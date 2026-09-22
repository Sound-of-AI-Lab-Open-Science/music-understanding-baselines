"""Build and train Music-JEPA with PyTorch Lightning.

This module is both a standalone trainer and the thing
``baselines/models/registry.py`` calls: its ``build(config)`` turns a config
dict into (datamodule, LightningModule), which is what makes a recipe in
``baselines/configs/`` and a hand-written config produce the same objects.
Normally you go through the scaffold instead:

    ../run_pretrain.sh jepa_paper [--smoke]
    python baselines/train.py --model jepa --config baselines/configs/jepa_paper.yaml ...

Pause / resume (lossless): training writes ``<out_dir>/checkpoints/last.ckpt``
every ``checkpoint_every_minutes`` (trainer YAML, default 10) of wall-clock time,
so a Ctrl-C / kill loses at most that interval.  Restart with::

    python scripts/train_jepa.py --config ... --resume            # auto: last.ckpt
    python scripts/train_jepa.py --config ... --resume-from PATH   # explicit ckpt

Lightning's ``ckpt_path`` restores the full training state -- model weights, the
EMA target-encoder weights (they live in the model state_dict), the AdamW moments,
the LR schedule, ``global_step``/epoch, and the dataloader position -- so training
continues exactly where it stopped (no loss/rho jump).

CLI overrides (handy for smoke runs / VRAM sweeps)::

    --data-root --max-steps --batch-size --seq-len --accum --devices --out-dir
    --resume --resume-from

YAML schema: ``data`` (JepaDataConfig fields), ``masking`` (MaskingConfig fields;
optional), ``model`` (MusicJepaConfig fields), ``loss`` (alpha/beta/sigma/gamma),
``optim`` (lr/weight_decay/warmup_ratio/ema_*), ``trainer`` (Lightning Trainer;
incl. ``checkpoint_every_minutes`` for the periodic last.ckpt cadence), plus a
top-level ``seed``.  Windows: ASCII logging only, ``PYTHONIOENCODING`` is forced
to utf-8, Rich progress bar disabled.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import yaml  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch  # noqa: E402
# Use the same Lightning package the JEPA datamodule inherits from
# (pytorch_lightning) so Trainer/Module/DataModule share one class tree.
try:
    import pytorch_lightning as L  # noqa: E402
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint  # noqa: E402
    from pytorch_lightning.loggers import CSVLogger  # noqa: E402
except Exception:  # pragma: no cover
    import lightning as L  # type: ignore # noqa: E402
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint  # noqa: E402
    from lightning.pytorch.loggers import CSVLogger  # noqa: E402

from src.data.jepa_datamodule import JepaDataModule, JepaDataConfig  # noqa: E402
from src.data.masking import MaskingConfig  # noqa: E402
from src.tasks.jepa_pretrain import MusicJepaLitModule  # noqa: E402


class AsciiLogCallback(Callback):
    """Plain-ASCII per-step logger with throughput + VRAM (Rich breaks on GBK)."""

    def __init__(self, every_n: int = 10):
        self.every_n = every_n
        self._t_last = None
        self._step_last = 0
        self._t_start = None
        self._step_start = 0
        self._total = None

    def on_train_start(self, trainer, pl_module):
        now = time.time()
        self._t_last = now
        self._step_last = trainer.global_step
        self._t_start = now
        self._step_start = trainer.global_step
        # Total optimizer steps for the whole run, so the log can print an ETA.
        try:
            total = int(trainer.estimated_stepping_batches)
        except Exception:
            total = -1
        if total is None or total <= 0:
            ms = int(getattr(trainer, "max_steps", -1) or -1)
            total = ms if ms > 0 else None
        self._total = total
        print("[train] total planned optimizer steps: {}".format(
            self._total if self._total else "unknown"), flush=True)

    @staticmethod
    def _fmt_hms(sec):
        if sec is None or sec != sec or sec < 0:
            return "?"
        sec = int(sec)
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return "{:d}h{:02d}m".format(h, m) if h else "{:d}m{:02d}s".format(m, s)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.every_n != 0 and step != 1:
            return
        now = time.time()
        dstep = max(1, step - self._step_last)
        sps = dstep / max(1e-9, (now - self._t_last))
        # Overall average since start -> stable ETA / percent.
        avg_sps = max(step - self._step_start, 1) / max(1e-9, now - self._t_start)
        if self._total:
            pct = 100.0 * step / self._total
            eta = (self._total - step) / max(1e-9, avg_sps)
            prog = "{:.1f}% {}/{} eta={}".format(
                pct, step, self._total, self._fmt_hms(eta))
        else:
            prog = "{}/? eta=?".format(step)
        self._t_last, self._step_last = now, step

        m = trainer.callback_metrics

        def g(k):
            v = m.get(k)
            try:
                return float(v)
            except Exception:
                return float("nan")

        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            mem_str = "{:.2f}GB".format(mem)
        else:
            mem_str = "n/a"
        print(
            "[{}] loss={:.4f} cos={:.4f} var={:.4f} cov={:.4f} "
            "jepa={:.4f} vicreg={:.3f} perdim_var={:.4f} lr={:.2e} "
            "rho={:.5f} {:.2f}it/s mem={}".format(
                prog, g("train/loss_step"), g("train/loss_cos_step"),
                g("train/loss_var_step"), g("train/loss_cov_step"),
                g("train/jepa_loss_step"), g("train/vicreg_loss_step"),
                g("train/perdim_var_step"), g("lr"), g("train/ema_rho"),
                sps, mem_str),
            flush=True)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


def _pkg_root_default(var: str, fallback: str) -> str:
    """A configured writable/readable root, not a directory inside the checkout.

    ``env.sh`` exports these and ``pkgpaths.py`` resolves the same values;
    importing ``pkgpaths`` fills them in when neither has run. The in-checkout
    fallback is last resort only and is covered by ``.gitignore``.
    """
    value = os.environ.get(var)
    if value:
        return value
    try:
        sys.path.insert(0, os.path.dirname(_PROJECT_ROOT))
        import pkgpaths  # noqa: F401  (resolution writes into os.environ)
        value = os.environ.get(var)
    except Exception:
        value = None
    return value or fallback


def build(config: dict):
    data_cfg = dict(config.get("data", {}))
    if "data_root" in data_cfg:
        data_cfg["data_root"] = _resolve(data_cfg["data_root"])
    dm = JepaDataModule(
        config=JepaDataConfig(**data_cfg),
        masking=MaskingConfig(**dict(config.get("masking", {}))) if config.get("masking") else None,
    )

    loss_cfg = dict(config.get("loss", {}))
    # The loss: section is consumed by an explicit whitelist below, not passed through.
    # A key that is present in the YAML but absent from that list is silently ignored and
    # the module takes its default, so a run can silently differ from what its config
    # says -- a control arm can end up byte-identical to the arm it controls for.
    # Fail loudly instead.
    _RECOGNISED_LOSS_KEYS = {
        "alpha", "beta", "gamma", "sigma", "var_eps", "cos_eps", "anti_collapse",
        "target_loss_norm", "repr_on_clean",
        "repr_std_weight", "repr_std_target", "repr_cov_weight",
        "repr_pool_weight", "repr_pool_cov_weight",
        "rank_reg_weight", "rank_reg_target", "rank_reg_bilateral",
        "equivar_weight", "equivar_max_semitones", "equivar_random_labels",
        "sigreg_lambda", "sigreg_num_slices", "sigreg_normalize",
        # two-view shift equivariance + encoder-side SIGReg
        "tv_equiv_weight", "tv_equiv_alpha", "tv_equiv_max_semitones",
        "tv_equiv_max_bars", "enc_sigreg_weight", "enc_sigreg_rows",
    }
    # A recognised key that never reaches the module is worse than a typo: the run
    # completes, exits 0, and reports nothing. Check the constructed module against
    # the config for every weight-like key.
    def _assert_forwarded(lit_obj, cfg):
        bad = []
        for k, v in cfg.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            got = getattr(lit_obj.hparams, k, None)
            if got is None or (float(v) != 0.0 and float(got) == 0.0):
                bad.append(f"{k}: config={v} module={got}")
        if bad:
            raise RuntimeError("loss keys not forwarded to the module:\n  " +
                               "\n  ".join(bad))

    _unknown = sorted(set(loss_cfg) - _RECOGNISED_LOSS_KEYS)
    if _unknown:
        raise SystemExit(
            f"config error: loss: keys {_unknown} are not read by train_jepa.py. "
            f"Add them to the module kwargs below AND to _RECOGNISED_LOSS_KEYS, or remove "
            f"them from the config -- do not leave them silently ignored.")
    optim_cfg = dict(config.get("optim", {}))
    lit = MusicJepaLitModule(
        model_config=config.get("model", {}),
        alpha=float(loss_cfg.get("alpha", 1.0)),
        beta=float(loss_cfg.get("beta", 25.0)),
        sigma=float(loss_cfg.get("sigma", 1.0)),
        gamma=float(loss_cfg.get("gamma", 1.0)),
        var_eps=float(loss_cfg.get("var_eps", 1e-4)),
        cos_eps=float(loss_cfg.get("cos_eps", 1e-8)),
        # anti-collapse family: "vicreg" (paper default) | "sigreg" (LeJEPA)
        anti_collapse=str(loss_cfg.get("anti_collapse", "vicreg")),
        target_loss_norm=bool(loss_cfg.get("target_loss_norm", False)),
        sigreg_lambda=float(loss_cfg.get("sigreg_lambda", 0.05)),
        sigreg_num_slices=int(loss_cfg.get("sigreg_num_slices", 1024)),
        sigreg_normalize=str(loss_cfg.get("sigreg_normalize", "per_n")),
        # anti-collapse: VICReg on the context representation (default 0 = off)
        repr_std_weight=float(loss_cfg.get("repr_std_weight", 0.0)),
        repr_cov_weight=float(loss_cfg.get("repr_cov_weight", 0.0)),
        repr_std_target=float(loss_cfg.get("repr_std_target", 1.0)),
        repr_pool_weight=float(loss_cfg.get("repr_pool_weight", 0.0)),
        repr_pool_cov_weight=float(loss_cfg.get("repr_pool_cov_weight", 0.0)),
        repr_on_clean=bool(loss_cfg.get("repr_on_clean", False)),
        # anti-collapse: soft effective-rank regularizer on pooled context (0 = off)
        rank_reg_weight=float(loss_cfg.get("rank_reg_weight", 0.0)),
        rank_reg_target=float(loss_cfg.get("rank_reg_target", 3.3)),
        rank_reg_bilateral=bool(loss_cfg.get("rank_reg_bilateral", True)),
        # Two-view equivariance + encoder-side SIGReg. The whitelist above only
        # catches misspelled keys; a key that is recognised but never forwarded here
        # falls back to the module default and the run silently does nothing -- which
        # is exactly what happened to the first smoke run (tv_equiv_weight recorded
        # as 0.0 in its checkpoint despite being 1.0 in the config).
        tv_equiv_weight=float(loss_cfg.get("tv_equiv_weight", 0.0)),
        tv_equiv_alpha=float(loss_cfg.get("tv_equiv_alpha", 0.3)),
        tv_equiv_max_semitones=int(loss_cfg.get("tv_equiv_max_semitones", 12)),
        tv_equiv_max_bars=int(loss_cfg.get("tv_equiv_max_bars", 8)),
        enc_sigreg_weight=float(loss_cfg.get("enc_sigreg_weight", 0.0)),
        enc_sigreg_rows=int(loss_cfg.get("enc_sigreg_rows", 2048)),
        equivar_weight=float(loss_cfg.get('equivar_weight', 0.0)),
        equivar_max_semitones=int(loss_cfg.get('equivar_max_semitones', 6)),
        equivar_random_labels=bool(loss_cfg.get('equivar_random_labels', False)),
        predictor_lr_scale=float(optim_cfg.get("predictor_lr_scale", 1.0)),
        lr=float(optim_cfg.get("lr", 1e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.05)),
        warmup_ratio=float(optim_cfg.get("warmup_ratio", 0.05)),
        adam_betas=tuple(optim_cfg.get("adam_betas", (0.9, 0.95))),
        adam_eps=float(optim_cfg.get("adam_eps", 1e-8)),
        min_lr_ratio=float(optim_cfg.get("min_lr_ratio", 0.0)),
        ema_rho_start=float(optim_cfg.get("ema_rho_start", 0.996)),
        ema_rho_end=float(optim_cfg.get("ema_rho_end", 1.0)),
        ema_mode=optim_cfg.get("ema_mode", "linear"),
        total_steps=optim_cfg.get("total_steps"),
    )
    _assert_forwarded(lit, loss_cfg)
    return dm, lit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None,
                    help="fix EMA/cosine schedule length (decouples it from a short "
                         "--max-steps smoke so early dynamics stay representative)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--devices", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="auto-resume from <out_dir>/checkpoints/last.ckpt if present "
                         "(else start fresh); restores weights/optimizer/LR/EMA/step.")
    ap.add_argument("--resume-from", default=None,
                    help="resume from an explicit checkpoint path (takes precedence "
                         "over --resume).")
    args = ap.parse_args()

    with open(_resolve(args.config), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.data_root:
        config.setdefault("data", {})["data_root"] = args.data_root
    if args.batch_size:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.seq_len:
        config.setdefault("data", {})["seq_len"] = args.seq_len
    if args.max_steps is not None:
        config.setdefault("trainer", {})["max_steps"] = args.max_steps
    if args.total_steps is not None:
        config.setdefault("optim", {})["total_steps"] = args.total_steps
    if args.accum is not None:
        config.setdefault("trainer", {})["accumulate_grad_batches"] = args.accum
    if args.devices is not None:
        config.setdefault("trainer", {})["devices"] = int(args.devices)

    seed = int(config.get("seed", 1))
    L.seed_everything(seed, workers=True)

    dm, lit = build(config)

    # Startup confirmation of the positional-encoding / relative-attention recipe
    # (so the log makes it unambiguous which model variant is training).
    _c = lit.cfg
    _abs_pe = _c.pos_encoding in ("absolute", "absolute_relative")
    _rel_attn = _c.pos_encoding in ("relative", "absolute_relative")
    print(
        "[train] model: pos_encoding={} (abs_PE={}, rel_attn={}"
        "{}) encoder_layers={} d_model={} predictor_layers={}".format(
            _c.pos_encoding, "ON" if _abs_pe else "off",
            "ON" if _rel_attn else "off",
            ", method={}".format(getattr(_c, "rel_attn_method", 2)) if _rel_attn else "",
            _c.encoder_layers, _c.d_model, _c.predictor_layers),
        flush=True)

    tr = dict(config.get("trainer", {}))
    out_dir = _resolve(args.out_dir or tr.pop(
        "default_root_dir",
        os.path.join(_pkg_root_default("RUN_ROOT", os.path.join(_PROJECT_ROOT, "runs")),
                     "jepa")))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    # Periodic checkpoint cadence is TIME-based (not step-based) so the "latest"
    # checkpoint refreshes at a fixed wall-clock rate regardless of model depth /
    # step throughput.  ``checkpoint_every_minutes`` (trainer YAML, default 10)
    # accepts floats (e.g. 0.25 == 15s for smoke tests).
    ckpt_every_min = float(tr.get("checkpoint_every_minutes", 10))

    logger = CSVLogger(save_dir=out_dir, name="", flush_logs_every_n_steps=10)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        AsciiLogCallback(every_n=int(tr.get("log_every_n_steps", 10))),
        # (1) PERIODIC "latest" checkpoint -- OWNS last.ckpt.  A TIME trigger plus
        # save_last=True rewrites last.ckpt every ``ckpt_every_min`` minutes with
        # the *current* training state, so a Ctrl-C / kill loses at most that
        # interval and ``--resume`` restores model + AdamW moments + LR schedule +
        # EMA target-encoder + global_step losslessly.  monitor=None keeps this
        # independent of val/loss -- the val monitor (callback 2) is exactly what
        # used to freeze last.ckpt at the best-val step (last.ckpt was only
        # rewritten on steps where val/loss improved), which broke resume.
        ModelCheckpoint(dirpath=ckpt_dir, monitor=None,
                        train_time_interval=timedelta(minutes=ckpt_every_min),
                        save_top_k=1, save_last=True,
                        filename="jepa-periodic-{step:06d}"),
        # (2) BEST-val checkpoint -- kept for model selection.  save_last=False so
        # it never competes with callback (1) for ownership of last.ckpt.
        ModelCheckpoint(dirpath=ckpt_dir, monitor="val/loss", mode="min",
                        save_top_k=1, save_last=False,
                        filename="jepa-best-{step:06d}"),
    ]
    # (3) OPTIONAL fork-finding: keep a *series* of step-based snapshots (never
    # overwritten) so the good/collapsed-basin divergence can be probed offline.
    # Off unless ``keep_ckpt_every_n_steps > 0``.  save_top_k=-1 keeps them all.
    keep_every = int(tr.get("keep_ckpt_every_n_steps", 0) or 0)
    if keep_every > 0:
        callbacks.append(
            ModelCheckpoint(dirpath=ckpt_dir, monitor=None,
                            every_n_train_steps=keep_every,
                            save_top_k=-1, save_last=False,
                            filename="jepa-keep-{step:06d}"))
    trainer = L.Trainer(
        max_steps=int(tr.get("max_steps", 300)),
        max_epochs=tr.get("max_epochs", -1),
        accelerator=tr.get("accelerator", "auto"),
        devices=tr.get("devices", 1),
        precision=tr.get("precision", "bf16-mixed"),
        accumulate_grad_batches=int(tr.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(tr.get("gradient_clip_val", 0.0)) or None,
        log_every_n_steps=int(tr.get("log_every_n_steps", 10)),
        val_check_interval=tr.get("val_check_interval", 1.0),
        limit_val_batches=tr.get("limit_val_batches", 1.0),
        num_sanity_val_steps=int(tr.get("num_sanity_val_steps", 0)),
        enable_progress_bar=tr.get("enable_progress_bar", False),
        logger=logger,
        callbacks=callbacks,
    )

    # -- resume: restore the FULL training state (weights/optimizer/LR/EMA/step) --
    # ckpt_path=None (the default when neither flag is given) is identical to the
    # legacy call, so behaviour is unchanged unless a resume flag is passed.
    ckpt_path = None
    if args.resume_from:
        ckpt_path = _resolve(args.resume_from)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                "--resume-from checkpoint not found: {}".format(ckpt_path))
        print("[train] --resume-from: restoring full training state from {}".format(
            ckpt_path), flush=True)
    elif args.resume:
        auto = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.isfile(auto):
            ckpt_path = auto
            print("[train] --resume: found {}, restoring full training state".format(
                auto), flush=True)
        else:
            print("[train] --resume: no checkpoint at {}; starting from scratch".format(
                auto), flush=True)

    trainer.fit(lit, datamodule=dm, ckpt_path=ckpt_path)

    final = os.path.join(out_dir, "music_jepa_final.pt")
    torch.save({"config": lit.cfg.to_dict(), "state_dict": lit.model.state_dict()}, final)
    print("[train] saved final model to {}".format(final), flush=True)


if __name__ == "__main__":
    main()
