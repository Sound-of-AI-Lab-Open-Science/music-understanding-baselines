"""LightningModule for Music-JEPA pre-training (Hachana & Rasheed).

Assembles the full training step:

    1. context encoder ``f_theta`` over ``context_tokens`` (with [MASK] subst);
    2. target encoder ``f_theta_EMA`` over the clean ``target_tokens`` under
       ``no_grad`` + ``eval`` (stop-gradient, no dropout);
    3. predictor emits ``yhat`` at the ``target_positions``;
    4. ``L_cos`` on the valid targets only, ``L_var`` / ``L_cov`` on the flattened
       predictor outputs (VICReg, applied to ``yhat`` per the paper);
    5. total ``= alpha*L_cos + beta*L_var + sigma*L_cov`` optimized with AdamW
       (lr 1e-4, wd 0.05, 5% warmup + cosine decay);
    6. EMA update of the target encoder once per optimizer step
       (``on_before_zero_grad``), ``rho`` following a 0.996 -> 1.0 schedule.

Everything the paper plots is logged: the two anti-phase curves ``train/jepa_loss``
(= ``alpha*L_cos``, Fig. 2 lower band ~0.2-1.0) and ``train/vicreg_loss``
(= ``beta*L_var + sigma*L_cov``, Fig. 2 upper band ~10-40), the raw terms, and
``train/perdim_var`` (Fig. 4, ~0-12) as the collapse monitor.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import torch

# Match the base class of src/data/jepa_datamodule.py, which prefers
# ``pytorch_lightning`` (a separate class tree from ``lightning.pytorch`` in this
# env); the Trainer, LightningModule and DataModule must all come from the same
# package or Lightning's ``is_overridden`` check raises "Expected a parent".
try:
    import pytorch_lightning as L
except Exception:  # pragma: no cover
    import lightning as L  # type: ignore

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.jepa import MusicJepa, MusicJepaConfig  # noqa: E402
from src.utils.ema import EmaMomentumSchedule, update_ema  # noqa: E402
from src.data.masking import MASK_TRANSPOSITION as M_TRANSPOSITION
from src.utils.vicreg import jepa_loss, variance_covariance, soft_effective_rank  # noqa: E402
from src.utils.sigreg import sigreg_jepa_loss  # noqa: E402



def _resolved_config_dict(cfg) -> dict:
    """Every field of the resolved config, as plain YAML-safe values.

    dataclasses.asdict is not used: it deep-copies and chokes on any field holding
    a tensor or a non-dataclass object, and a persistence helper that can raise
    would take the training run down with it. Anything unrepresentable is stored
    as its repr rather than dropped, because a wrong-looking value in the record
    is recoverable and a missing field is what caused this bug.
    """
    out = {}
    for k in getattr(cfg, "__dataclass_fields__", {}):
        v = getattr(cfg, k, None)
        if isinstance(v, (bool, int, float, str, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (bool, int, float, str, type(None)))
                      else repr(x) for x in v]
        elif isinstance(v, dict):
            out[k] = {str(a): (b if isinstance(b, (bool, int, float, str, type(None)))
                               else repr(b)) for a, b in v.items()}
        else:
            out[k] = repr(v)
    return out

class MusicJepaLitModule(L.LightningModule):
    def __init__(
        self,
        model_config: Optional[dict] = None,
        # loss weights (paper §3.3; alpha=gamma=1 fixed, beta/sigma calibrated)
        alpha: float = 1.0,
        beta: float = 25.0,
        sigma: float = 1.0,
        gamma: float = 1.0,
        var_eps: float = 1e-4,
        cos_eps: float = 1e-8,
        # -- anti-collapse family selector: "vicreg" (paper default) | "sigreg" --
        # "vicreg" keeps the paper's beta*L_var + sigma*L_cov on yhat (bit-for-bit
        # unchanged).  "sigreg" replaces those terms with LeJEPA's SIGReg
        # (Balestriero & LeCun, arXiv:2511.08544): the objective becomes
        # (1 - sigreg_lambda)*L_cos + sigreg_lambda*mean(SIGReg(yhat)), and the
        # repr_* extra VICReg terms below are BYPASSED.  Default "vicreg" so the
        # already-running faithful/ablation experiments are untouched.
        anti_collapse: str = "vicreg",
        # I-JEPA-style target normalization: LayerNorm the TARGET features on the
        # fly INSIDE the loss (target branch only, per-token over the feature dim,
        # exactly as official facebookresearch/ijepa does before its smooth-L1).
        # Pair with model.encoder_final_norm: false -> the persistent
        # representation stays free-scale while the regression target is still
        # scale-stabilized. Default False = existing behaviour untouched.
        target_loss_norm: bool = False,
        sigreg_lambda: float = 0.05,
        sigreg_num_slices: int = 1024,
        sigreg_normalize: str = "per_n",
        # -- anti-collapse: VICReg on the CONTEXT REPRESENTATION (not just yhat) --
        # The paper applies var/cov to the predictor output ``yhat`` only, which is
        # decoupled from the context encoder features used downstream and cannot
        # stop the encoder collapsing every token to one shared vector (observed
        # as token_cos -> 1.0 with per_dim_var -> 0).
        # These weights add a variance-hinge + covariance penalty directly on the
        # context representation at the real/visible positions.  Default 0.0 ->
        # behaviour is bit-for-bit the paper recipe; > 0 turns the fix on.
        repr_std_weight: float = 0.0,
        repr_cov_weight: float = 0.0,
        repr_std_target: float = 1.0,   # gamma of the repr variance hinge
        # -- anti-collapse: VICReg on the SONG-LEVEL (mean-pooled) representation --
        # Token-level repr-VICReg centers over the whole token population, so it is
        # BLIND to a shared per-song offset -- observed as the residual failure: a
        # run that held token std=1 still had pooled_cos 0.99 and random retrieval.
        # This variance hinge (+ optional covariance) acts on the per-sample mean-
        # pooled clean representation across the batch, directly forcing DIFFERENT
        # SONGS apart -- the exact quantity retrieval ranks on.  Requires
        # repr_on_clean=True (needs the clean per-sample encoding).  Default 0 = off.
        repr_pool_weight: float = 0.0,
        repr_pool_cov_weight: float = 0.0,
        # Where to measure/regularize the context representation.  The collapse is
        # of the CLEAN full-sequence encoding (what downstream uses); the masked
        # context stays spread even when collapsed (diagnosed: masked repr_std ~0.25
        # while clean per_dim_var ~0).  repr_on_clean=True runs one extra context-
        # encoder pass on the clean target_tokens (with grad) and regularizes THAT
        # -- the correct target -- at the cost of a second encoder forward/backward.
        # Default False keeps the cheap masked-view monitor and the paper's cost.
        repr_on_clean: bool = False,
        # -- anti-collapse: SOFT EFFECTIVE-RANK regularizer on POOLED context -----
        # The residual collapse lives in the per-song mean-pooled context features:
        # their effective rank (exp of the Shannon entropy of the normalized
        # eigenvalue spectrum of the feature covariance) is ~1.4 in collapsed runs
        # vs ~3.3 in healthy ones.  This differentiable soft-rank penalty (IDRR /
        # Matrix-SSL style) pushes the batch pooled-feature spectrum back toward the
        # healthy target so the encoder keeps multiple active directions.  Backward
        # flows into the encoder through the pooled features.  Default weight 0.0 ->
        # EXACT no-op (the term is skipped entirely, loss is bit-for-bit the recipe).
        rank_reg_weight: float = 0.0,
        rank_reg_target: float = 3.3,     # target effective rank
        rank_reg_bilateral: bool = True,  # penalize both sides of target (else only below)
        # -- EQUIVARIANCE auxiliary head (transposition-shift prediction) -------
        # The paper's FME injects transposition INVARIANCE into the embedding; our
        # ablation shows that destroys the retrieval basin (cf. LOEV 2024: baking
        # invariance into the objective wrecks pitch-dependent tasks).  The
        # equivariant alternative is to KEEP pitch-shift information and make it
        # RECOVERABLE: on transposition-masked samples (context = transposed,
        # target = original) a small head predicts the shift k from the pair of
        # pooled representations.  Default weight 0.0 -> exact no-op.
        # YAML: loss.equivar_weight / loss.equivar_max_semitones
        # -- MIDI-RAE-style two-view shift equivariance as a PRIMARY term ------
        # Distinct from `equivar_weight` below, which is a parameterised classifier
        # head fired on 1/7 of samples. This one is parameter-free, fires on every
        # sample every step, and both attracts and repels (see
        # src/utils/equivariance.py). alpha must be calibrated on a smoke run --
        # the paper does not publish it.
        tv_equiv_weight: float = 0.0,
        tv_equiv_alpha: float = 0.3,
        tv_equiv_max_semitones: int = 12,
        tv_equiv_max_bars: int = 8,
        enc_sigreg_weight: float = 0.0,
        enc_sigreg_rows: int = 2048,
        equivar_weight: float = 0.0,
        equivar_max_semitones: int = 6,   # classes = 2*max+1 (shift -max..+max)
        # Control arm: keep the head and its gradient, replace the target with an
        # arbitrary fixed label carrying no transposition information. Falsifies
        # "the equivariance objective specifically" against "any aux head that
        # needs discriminable pooled features". Default off = existing behaviour.
        equivar_random_labels: bool = False,
        # optimizer (I-JEPA/BERT-style; AdamW)
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.05,
        adam_betas=(0.9, 0.95),
        adam_eps: float = 1e-8,
        min_lr_ratio: float = 0.0,
        # predictor LR scale (<1 handicaps the predictor so it cannot as easily
        # find the trivial "scale one constant direction" cheat; 1.0 = unchanged).
        predictor_lr_scale: float = 1.0,
        # EMA schedule
        ema_rho_start: float = 0.996,
        ema_rho_end: float = 1.0,
        ema_mode: str = "linear",
        # total optimizer steps (None -> trainer.estimated_stepping_batches)
        total_steps: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        cfg = MusicJepaConfig.from_dict(model_config or {})
        self.cfg = cfg
        # T30 -- persist the RESOLVED config, not just what the YAML happened to say.
        # save_hyperparameters() captures the constructor arguments, and
        # `model_config` is the raw dict; every field the YAML omitted is filled from
        # the dataclass defaults inside from_dict() and never reaches hparams.yaml.
        # A survey on 2026-08-28 found `use_fme` absent from 1016 of 1154 hparams.yaml
        # files for exactly this reason -- which is why arm comparisons kept seeing
        # `None` and why the witness guard in tools/make_comparison.py has to fall
        # back on weight shapes. Writing the resolved config removes the ambiguity at
        # the source; the guard stays as the check for runs predating this line.
        self.hparams["resolved_model_config"] = _resolved_config_dict(cfg)
        self.model = MusicJepa(cfg)
        # convenient aliases for the EMA machinery / callback
        self.context_encoder = self.model.context_encoder
        self.target_encoder = self.model.target_encoder
        # Equivariance head: [pooled_transposed ; pooled_original] -> shift class.
        # Built only when the term is on, so weight=0 runs keep an identical
        # state_dict (checkpoint-compatible with every earlier run).
        if float(equivar_weight) > 0.0:
            n_cls = 2 * int(equivar_max_semitones) + 1
            self.equivar_head = torch.nn.Sequential(
                torch.nn.Linear(2 * cfg.d_model, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, n_cls),
            )

        self.ema_schedule = EmaMomentumSchedule(
            rho_start=ema_rho_start, rho_end=ema_rho_end,
            total_steps=total_steps, mode=ema_mode,
        )
        self._last_rho = ema_rho_start

    # -- forward / loss ------------------------------------------------------
    def _step_loss(self, batch):
        # Always request the context representation: it is free (already computed
        # in forward) and lets us (a) MONITOR its per-dim std -- the number that
        # actually detects the constant-collapse -- and (b) optionally regularize
        # it (repr_std_weight / repr_cov_weight).
        yhat, y, valid, context = self.model(batch, return_context=True)
        yv, yvt = self.model.flatten_valid(yhat, y, valid)
        if bool(getattr(self.hparams, "target_loss_norm", False)) and yvt.shape[0] > 0:
            # I-JEPA-style: normalize the regression TARGET on the fly (no grad
            # flows through it anyway; the persistent representation is untouched).
            yvt = torch.nn.functional.layer_norm(yvt, (yvt.shape[-1],))
        use_sigreg = str(self.hparams.anti_collapse).lower() == "sigreg"
        if yv.shape[0] == 0:  # degenerate guard (never happens: >=1 target/sample)
            zero = yhat.sum() * 0.0
            out = {"loss": zero, "loss_cos": zero, "loss_var": zero, "loss_cov": zero,
                   "jepa_loss": zero, "vicreg_loss": zero,
                   "per_dim_variance": zero, "mean_std": zero,
                   "repr_std": zero, "repr_perdim_var": zero, "repr_reg": zero,
                   "repr_pool_std": zero,
                   "soft_rank": zero, "rank_reg_loss": zero,
                   "equivar_loss": zero, "equivar_acc": zero}
            if use_sigreg:
                out["sigreg_loss"] = zero
                out["sigreg_raw"] = zero
            return out
        if use_sigreg:
            # LeJEPA path: SIGReg replaces VICReg var/cov (arXiv:2511.08544).  The
            # repr_* extra VICReg terms are bypassed entirely (off in the faithful
            # config anyway).  Provide loss_var/loss_cov/vicreg_loss as zeros so the
            # existing logging in training_step/validation_step stays valid.
            out = sigreg_jepa_loss(
                yv, yvt,
                lambd=float(self.hparams.sigreg_lambda),
                global_step=int(self.global_step),
                cos_eps=self.hparams.cos_eps,
                num_slices=int(self.hparams.sigreg_num_slices),
                normalize=str(self.hparams.sigreg_normalize),
                var_eps=self.hparams.var_eps,
            )
            zero = yhat.sum() * 0.0
            out.setdefault("loss_var", zero)
            out.setdefault("loss_cov", zero)
            out.setdefault("vicreg_loss", out["sigreg_loss"])
            out.setdefault("repr_std", zero)
            out.setdefault("repr_perdim_var", zero)
            out.setdefault("repr_reg", zero)
            out.setdefault("repr_pool_std", zero)
            out.setdefault("soft_rank", zero)
            out.setdefault("rank_reg_loss", zero)
            out.setdefault("equivar_loss", zero)
            out.setdefault("equivar_acc", zero)
            return out
        out = jepa_loss(
            yv, yvt,
            alpha=self.hparams.alpha, beta=self.hparams.beta,
            sigma=self.hparams.sigma, gamma=self.hparams.gamma,
            var_eps=self.hparams.var_eps, cos_eps=self.hparams.cos_eps,
        )

        # -- context-representation statistics + optional regularizer ----------
        w_std = float(self.hparams.repr_std_weight)
        w_cov = float(self.hparams.repr_cov_weight)
        w_pool = float(self.hparams.repr_pool_weight)
        w_pool_cov = float(self.hparams.repr_pool_cov_weight)
        clean_ctx = None
        if bool(self.hparams.repr_on_clean):
            # Regularize/monitor the CLEAN full-sequence encoding (extra context
            # pass, with grad) -- the exact tensor downstream consumes.  Use all
            # real positions.
            clean_ctx = self.model.context_encoder(
                batch["target_tokens"], pad_mask=batch["pad_mask"], visible_mask=None)
            cf = clean_ctx[batch["pad_mask"]]                    # [M,d]
        else:
            # Cheap monitor on REAL & VISIBLE positions of the masked context
            # (excludes padding and the [MASK] placeholder constant).
            vis = batch["context_mask"] & batch["pad_mask"]      # [B,S]
            cf = context[vis]                                    # [M,d]
        gamma = float(self.hparams.repr_std_target)
        reg = context.sum() * 0.0
        if cf.shape[0] >= 2:
            vc = variance_covariance(cf, gamma=gamma, eps=self.hparams.var_eps)
            out["repr_std"] = vc["mean_std"]
            out["repr_perdim_var"] = vc["per_dim_variance"]
            if w_std > 0.0 or w_cov > 0.0:
                reg = reg + w_std * vc["loss_var"] + w_cov * vc["loss_cov"]
        else:
            out["repr_std"] = context.sum() * 0.0
            out["repr_perdim_var"] = context.sum() * 0.0

        # -- song-level (mean-pooled) variance: forces DIFFERENT SONGS apart -----
        out["repr_pool_std"] = out["repr_std"] * 0.0
        if clean_ctx is not None and (w_pool > 0.0 or w_pool_cov > 0.0):
            pad = batch["pad_mask"]
            cnt = pad.sum(1, keepdim=True).clamp(min=1)
            pooled = (clean_ctx * pad.unsqueeze(-1)).sum(1) / cnt    # [B,d] song means
            if pooled.shape[0] >= 2:
                vcp = variance_covariance(pooled, gamma=gamma, eps=self.hparams.var_eps)
                out["repr_pool_std"] = vcp["mean_std"].detach()
                reg = reg + w_pool * vcp["loss_var"] + w_pool_cov * vcp["loss_cov"]

        # -- soft effective-rank regularizer on the POOLED context features ------
        # Mean-pool the context representation over valid (real) positions -> one
        # [B, d] vector per song, then push its batch soft effective rank toward the
        # healthy target.  Uses the clean encoding when repr_on_clean has already
        # produced it (the exact tensor downstream consumes); otherwise the masked
        # context (its pooled rank tracks the same collapse, just noisier).
        out["soft_rank"] = context.sum() * 0.0
        out["rank_reg_loss"] = context.sum() * 0.0
        w_rank = float(getattr(self.hparams, "rank_reg_weight", 0.0))
        if w_rank > 0.0:
            src = clean_ctx if clean_ctx is not None else context   # [B,S,d]
            pad = batch["pad_mask"]                                  # [B,S] True=real
            cnt = pad.sum(1, keepdim=True).clamp(min=1)
            pooled = (src * pad.unsqueeze(-1)).sum(1) / cnt          # [B,d]
            if pooled.shape[0] >= 4:                                 # guard tiny batches
                sr = soft_effective_rank(pooled, eps=1e-8)          # float32 scalar
                out["soft_rank"] = sr.detach()
                target = float(self.hparams.rank_reg_target)
                if bool(self.hparams.rank_reg_bilateral):
                    dev = sr - target                               # both sides
                else:
                    dev = torch.clamp(target - sr, min=0.0)          # only below
                rank_reg = w_rank * dev.pow(2)
                reg = reg + rank_reg
                out["rank_reg_loss"] = rank_reg.detach()

        # -- TWO-VIEW EQUIVARIANCE (primary) + encoder-side SIGReg ---------------
        # The paper's VICReg only touches yhat (jepa.py:15) and cannot constrain the
        # encoder; encoder-side VICReg hinges were tried and destroyed the basin
        # (reprvic6L 0/3 vs base 6/6). This applies SIGReg where LeJEPA applies it --
        # the student encoder's embeddings -- and adds the shift-equivariance term
        # that MIDI-RAE-JEPA credits for its result.
        w_tv = float(getattr(self.hparams, "tv_equiv_weight", 0.0))
        w_es = float(getattr(self.hparams, "enc_sigreg_weight", 0.0))
        if w_tv > 0.0 or w_es > 0.0:
            from src.utils.equivariance import (sample_shifts, shift_view,
                                                masked_mean_pool, equivariance_loss)
            pad = batch["pad_mask"]
            g = torch.Generator(device=batch["target_tokens"].device)
            g.manual_seed(int(self.global_step) * 7919 + 13)
            dp, db = sample_shifts(pad.shape[0],
                                   int(self.hparams.tv_equiv_max_semitones),
                                   int(self.hparams.tv_equiv_max_bars),
                                   g, batch["target_tokens"].device)
            shifted, dp, db = shift_view(batch["target_tokens"], pad, dp, db)
            h2 = self.model.context_encoder(shifted, pad_mask=pad, visible_mask=None)
            z2 = masked_mean_pool(h2, pad)
            if w_tv > 0.0:
                z1 = masked_mean_pool(self.model.encode_target(batch).detach(), pad)
                eqv, dist, tgt = equivariance_loss(
                    z1, z2, dp, db,
                    int(self.hparams.tv_equiv_max_semitones),
                    int(self.hparams.tv_equiv_max_bars),
                    float(self.hparams.tv_equiv_alpha))
                reg = reg + w_tv * eqv
                out["tv_equiv"] = eqv.detach()
                out["tv_dist_med"] = dist.median().detach()
                out["tv_target_med"] = tgt.median().detach()
            if w_es > 0.0:
                from src.utils.sigreg import sigreg_term
                rows = h2[pad]
                cap = int(self.hparams.enc_sigreg_rows)
                if rows.shape[0] > cap:
                    idx = torch.randperm(rows.shape[0], generator=g,
                                         device=rows.device)[:cap]
                    rows = rows[idx]
                es = sigreg_term(rows, global_step=int(self.global_step),
                                 num_slices=int(self.hparams.sigreg_num_slices),
                                 normalize="raw")
                es = es + sigreg_term(z2, global_step=int(self.global_step) + 1,
                                      num_slices=int(self.hparams.sigreg_num_slices),
                                      normalize="raw")
                reg = reg + w_es * es
                out["enc_sigreg"] = es.detach()

        # -- EQUIVARIANCE auxiliary: recover the transposition shift --------------
        # Active only on transposition-masked samples (mask type 6), where the
        # context IS the transposed sequence and target_tokens the original.  The
        # head sees [pool(context) ; pool(original)] and classifies the shift, so
        # the encoder is pushed to make pitch-shift RECOVERABLE (equivariant)
        # instead of discarding it (what FME's invariance does).
        zero = context.sum() * 0.0
        out["equivar_loss"] = zero
        out["equivar_acc"] = zero
        w_eq = float(getattr(self.hparams, "equivar_weight", 0.0))
        if w_eq > 0.0 and hasattr(self, "equivar_head"):
            is_tr = batch["latent_mask_type"] == M_TRANSPOSITION          # [B]
            if bool(is_tr.any()):
                if clean_ctx is None:
                    clean_ctx = self.model.context_encoder(
                        batch["target_tokens"], pad_mask=batch["pad_mask"],
                        visible_mask=None)
                pad = batch["pad_mask"]
                cnt = pad.sum(1, keepdim=True).clamp(min=1)
                p_ctx = (context * pad.unsqueeze(-1)).sum(1) / cnt        # [B,d] transposed
                p_org = (clean_ctx * pad.unsqueeze(-1)).sum(1) / cnt      # [B,d] original
                feats = torch.cat([p_ctx[is_tr], p_org[is_tr]], dim=-1)   # [n,2d]
                logits = self.equivar_head(feats)
                mx = int(self.hparams.equivar_max_semitones)
                if bool(getattr(self.hparams, "equivar_random_labels", False)):
                    # CONTROL ARM. Same head, same gradient path, same number of classes,
                    # but the target carries no transposition information -- it is an
                    # arbitrary class derived from a content hash of target_tokens.
                    #
                    # Why a *fixed* pseudo-random label and not a freshly drawn one: the
                    # hypothesis under test is that the equivariance objective raises
                    # per-dim variance / lowers pairwise cosine only because classifying
                    # from pooled features REQUIRES those features to be discriminable.
                    # A fresh draw each step is unlearnable, sits at ln(13), and so does
                    # not demand discriminability at all -- it would control for "extra
                    # gradient noise", which is not the concern. A fixed arbitrary label
                    # can only be fitted by memorising, which demands exactly the same
                    # discriminability while encoding nothing about pitch shift.
                    #
                    # target_tokens is the ORIGINAL (untransposed) sequence, so the label
                    # is stable across epochs AND independent of the shift by construction.
                    tt = batch["target_tokens"][is_tr].long()             # [n, L, F]
                    pm = batch["pad_mask"][is_tr].long()                  # [n, L]
                    w = (torch.arange(1, tt.shape[-1] + 1, device=tt.device,
                                      dtype=torch.long) * 2654435761)
                    h = ((tt * w).sum(-1) * pm).sum(-1)                   # [n]
                    cls = (h % (2 * mx + 1)).clamp(0, 2 * mx)             # [n]
                else:
                    shift = batch["latent_params"][is_tr, 0]              # semitones
                    cls = (shift.round().long() + mx).clamp(0, 2 * mx)    # [n]
                eq = torch.nn.functional.cross_entropy(logits.float(), cls)
                reg = reg + w_eq * eq
                out["equivar_loss"] = eq.detach()
                out["equivar_acc"] = (logits.argmax(-1) == cls).float().mean().detach()

        out["loss"] = out["loss"] + reg
        out["repr_reg"] = reg.detach() if torch.is_tensor(reg) else reg
        return out

    def training_step(self, batch, batch_idx):
        out = self._step_loss(batch)
        bs = batch["context_tokens"].shape[0]
        log = dict(on_step=True, on_epoch=True, batch_size=bs)
        self.log("train/loss", out["loss"], prog_bar=True, **log)
        self.log("train/loss_cos", out["loss_cos"], **log)
        if float(getattr(self.hparams, "equivar_weight", 0.0)) > 0.0:
            self.log("train/equivar_loss", out["equivar_loss"], **log)
            self.log("train/equivar_acc", out["equivar_acc"], prog_bar=True, **log)
        self.log("train/loss_var", out["loss_var"], **log)
        self.log("train/loss_cov", out["loss_cov"], **log)
        self.log("train/jepa_loss", out["jepa_loss"], prog_bar=True, **log)
        self.log("train/vicreg_loss", out["vicreg_loss"], prog_bar=True, **log)
        if "sigreg_loss" in out:
            self.log("train/sigreg_loss", out["sigreg_loss"], prog_bar=True, **log)
        self.log("train/perdim_var", out["per_dim_variance"], prog_bar=True, **log)
        self.log("train/mean_std", out["mean_std"], **log)
        # context-representation collapse monitor (the real early-warning signal:
        # this is the tensor downstream consumes; constant-collapse drives it -> 0)
        self.log("train/repr_std", out["repr_std"], prog_bar=True, **log)
        self.log("train/repr_perdim_var", out["repr_perdim_var"], **log)
        self.log("train/repr_reg", out["repr_reg"], **log)
        self.log("train/repr_pool_std", out["repr_pool_std"], prog_bar=True, **log)
        # soft effective-rank monitor + its regularizer contribution
        self.log("train/soft_rank", out["soft_rank"], prog_bar=True, **log)
        self.log("train/rank_reg_loss", out["rank_reg_loss"], **log)
        self.log("lr", self._current_lr(), prog_bar=True, on_step=True)
        # Two-view equivariance / encoder SIGReg: only present when enabled, so log
        # conditionally. Without this the smoke run computes them and records
        # nothing, which is how alpha calibration was missed the first time.
        for k, pb in (("tv_equiv", True), ("tv_dist_med", True), ("tv_target_med", True),
                      ("enc_sigreg", True)):
            if k in out:
                self.log(f"train/{k}", out[k], prog_bar=pb, **log)
        self.log("train/ema_rho", self._last_rho, on_step=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._step_loss(batch)
        bs = batch["context_tokens"].shape[0]
        log = dict(on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("val/loss", out["loss"], prog_bar=True, **log)
        self.log("val/loss_cos", out["loss_cos"], **log)
        self.log("val/loss_var", out["loss_var"], **log)
        self.log("val/loss_cov", out["loss_cov"], **log)
        self.log("val/perdim_var", out["per_dim_variance"], **log)
        self.log("val/repr_std", out["repr_std"], **log)
        return out["loss"]

    # -- EMA update: once per optimizer step (handles grad accumulation) ------
    def on_before_zero_grad(self, optimizer):
        rho = self.ema_schedule.value(int(self.trainer.global_step))
        self._last_rho = rho
        update_ema(self.model.target_encoder, self.model.context_encoder, rho)

    # -- keep the datamodule's per-epoch masking seed in sync ---------------
    def on_train_epoch_start(self):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "set_epoch"):
            dm.set_epoch(self.current_epoch)

    # -- optimizer + warmup/cosine schedule ---------------------------------
    def _resolve_total_steps(self) -> int:
        if self.hparams.total_steps:
            return int(self.hparams.total_steps)
        try:
            return max(1, int(self.trainer.estimated_stepping_batches))
        except Exception:
            return 100000

    def configure_optimizers(self):
        total = self._resolve_total_steps()
        self.ema_schedule.set_total_steps(total)
        warmup = max(1, int(round(self.hparams.warmup_ratio * total)))

        # Split params into 4 groups: {predictor, rest} x {decay, no_decay}.  The
        # predictor groups get lr * predictor_lr_scale (LambdaLR multiplies each
        # group's own base lr by the same schedule factor, so the ratio holds).
        pred_scale = float(self.hparams.predictor_lr_scale)
        base_lr = self.hparams.lr
        buckets = {"pred_decay": [], "pred_no_decay": [], "decay": [], "no_decay": []}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_pred = name.startswith("predictor.")
            is_no_decay = (p.ndim == 1 or name.endswith(".bias"))
            key = ("pred_" if is_pred else "") + ("no_decay" if is_no_decay else "decay")
            buckets[key].append(p)
        groups = [
            {"params": buckets["decay"], "weight_decay": self.hparams.weight_decay,
             "lr": base_lr},
            {"params": buckets["no_decay"], "weight_decay": 0.0, "lr": base_lr},
            {"params": buckets["pred_decay"], "weight_decay": self.hparams.weight_decay,
             "lr": base_lr * pred_scale},
            {"params": buckets["pred_no_decay"], "weight_decay": 0.0,
             "lr": base_lr * pred_scale},
        ]
        groups = [g for g in groups if g["params"]]  # drop empties
        optimizer = torch.optim.AdamW(
            groups, lr=self.hparams.lr,
            betas=tuple(self.hparams.adam_betas), eps=self.hparams.adam_eps,
        )

        min_ratio = float(self.hparams.min_lr_ratio)

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            prog = (step - warmup) / max(1, total - warmup)
            prog = min(prog, 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))       # 1 -> 0
            return min_ratio + (1.0 - min_ratio) * cos

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _current_lr(self):
        opts = self.optimizers()
        if isinstance(opts, (list, tuple)):
            opts = opts[0]
        try:
            return opts.param_groups[0]["lr"]
        except Exception:
            return float("nan")
