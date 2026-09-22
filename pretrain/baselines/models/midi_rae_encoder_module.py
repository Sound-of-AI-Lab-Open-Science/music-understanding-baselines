"""LightningModule port of midi-rae's encoder training loop
(``pretrain/midi_rae/midi_rae/train_enc.py::train`` + ``compute_batch_loss``,
now removed as a standalone script/pipeline -- this is the integrated
replacement).

This is a REAL behavioral port, not a thin wrapper: the EMA teacher update,
the per-level LeJEPA (SIGReg + attraction + factorization) loss, and the MEP
(masked embedding prediction) loss are all reproduced here rather than called
into a hand-rolled loop, because upstream trains with one (there is no
``model.compute_loss``-style entry point to call the way MuseTok's wrapper
does -- see ``musetok_module.py`` for that pattern and how it differs).

Preserved from the two LOCAL modifications on top of vanilla upstream that
this port must keep faithful to:
  * the gradual log-space EMA eta ramp (``ema_eta_ramp_epochs`` > 0 interpolates
    log10(eta) linearly from ``ema_eta`` to ``ema_eta_after`` over that many
    epochs after ``ema_eta_switch_epoch``), replacing a hard jump;
  * sharded, lazily-loaded piano-roll training (via ``ShardedTripletDataset``,
    wrapped by ``baselines/data/pianoroll_datamodule.py::PianorollTripletDataModule``).

Only the swin encoder path is ported (see ``midi_rae_lib/__init__.py``): both
midi_rae_enc.yaml and midi_rae_dec.yaml set ``model.encoder: swin`` and
``training.lambda_mae: 0.0``, so the MAE-decoder branch of upstream's
``compute_batch_loss`` never executes and is not reproduced here.

Verification level: a CPU smoke test only (tiny synthetic shards, a handful of
steps) proves the wiring -- right shapes, gradients flow, finite losses. It
does NOT establish long-run numerical equivalence with the original hand-written
loop; treat any specific loss trajectory / convergence comparison as unverified
until a GPU run is checked against the upstream script's numbers.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import lightning as L
import torch

from baselines.models.midi_rae_lib.core import HierarchicalPatchState
from baselines.models.midi_rae_lib.ema import EMAModel
from baselines.models.midi_rae_lib.losses import calc_enc_loss_multiscale
from baselines.models.midi_rae_lib.swin import SwinEncoder, SwinMaskedEmbeddingPredictor


class MidiRaeEncoderModule(L.LightningModule):
    """Swin encoder trained with LeJEPA (SIGReg + attraction + factorization) + MEP."""

    def __init__(self, model: Optional[dict] = None, training: Optional[dict] = None,
                 image_size: int = 128):
        super().__init__()
        self.save_hyperparameters()
        m = dict(model or {})
        tr = dict(training or {})
        self.image_size = int(image_size)

        depths = tuple(m.get('depths', (2, 2, 2, 6, 2, 1)))
        num_heads = tuple(m.get('num_heads', (2, 2, 2, 4, 8, 16)))
        embed_dim = int(m.get('embed_dim', 8))
        self.encoder = SwinEncoder(
            img_height=image_size, img_width=image_size,
            patch_h=int(m.get('patch_h', 4)), patch_w=int(m.get('patch_w', 4)),
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=int(m.get('window_size', 4)), mlp_ratio=float(m.get('mlp_ratio', 4.0)),
            drop_path_rate=float(m.get('drop_path_rate', 0.1)))

        self.lambda_mep = float(tr.get('lambda_mep', 1.0))
        self.mep_model = None
        if self.lambda_mep > 0:
            dims = tuple(embed_dim * 2 ** i for i in range(len(depths) - 1, -1, -1))
            self.mep_model = SwinMaskedEmbeddingPredictor(dims=dims)

        ema_eta = float(tr.get('ema_eta', 1e-5))
        self.ema_encoder = EMAModel(self.encoder, eta=ema_eta,
                                    update_every=int(tr.get('ema_update_every', 1)),
                                    dtype=torch.float32) if ema_eta > 0 else None

        self.tr = tr
        # updated once per epoch (see on_train_epoch_start), mirrors upstream's
        # in-loop ``loss_weights`` dict that only ``lambda_sim`` mutates per-epoch.
        self._lambda_sim_epochs = int(tr.get('lambda_sim_curriculum_epochs', 0))
        self._loss_weights = {
            'lambd': float(tr.get('lambd', 0.5)),
            'lambda_fact': float(tr.get('lambda_fact', 0.5)),
            'lambda_anchor': float(tr.get('lambda_anchor', 0.0)),
            'lambda_sim': 0.0 if self._lambda_sim_epochs > 0 else float(tr.get('lambda_sim', 1.0)),
            'lambda_mep': self.lambda_mep,
            'lambda_mae': float(tr.get('lambda_mae', 0.0)),
        }

    # -- EMA eta ramp (LOCAL modification, preserved: gradual log-space ramp,
    # not a hard jump) ------------------------------------------------------
    def on_train_epoch_start(self):
        epoch = self.current_epoch + 1  # upstream epochs are 1-indexed
        if self._lambda_sim_epochs > 0:
            self._loss_weights['lambda_sim'] = min(1.0, (epoch - 1) / self._lambda_sim_epochs)

        if self.ema_encoder is None:
            return
        switch_epoch = int(self.tr.get('ema_eta_switch_epoch', 44))
        ramp_epochs = int(self.tr.get('ema_eta_ramp_epochs', 0))
        if epoch < switch_epoch:
            return
        eta_after = float(self.tr.get('ema_eta_after', 0.96))
        if ramp_epochs > 0:
            eta_start = max(float(self.tr.get('ema_eta', 1e-5)), 1e-12)
            frac = min(1.0, (epoch - switch_epoch) / ramp_epochs)
            log_eta = math.log10(eta_start) + frac * (math.log10(eta_after) - math.log10(eta_start))
            self.ema_encoder.eta = 10 ** log_eta
        else:
            self.ema_encoder.eta = eta_after

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_encoder is not None:
            self.ema_encoder.update(self.encoder)

    # -- loss (ported from compute_batch_loss) -------------------------------
    def _compute_loss(self, batch, stage: str):
        img1, img2, deltas = batch['img1'], batch['img2'], batch['deltas']
        img3 = batch.get('img3')
        target = batch.get('target')
        lw = self._loss_weights

        enc1_fn = self.ema_encoder if self.ema_encoder is not None else self.encoder
        enc_out1 = enc1_fn(img1, mask_ratio=0)
        enc_out2 = self.encoder(img2, mae_mask=enc_out1.mae_mask)
        enc_out3 = None
        if img3 is not None and lw.get('lambda_fact', 0) > 0:
            enc_out3 = self.encoder(img3, mae_mask=enc_out1.mae_mask)

        assert isinstance(enc_out1.patches, HierarchicalPatchState)
        z1 = [lvl.emb for lvl in enc_out1.patches.levels]
        z2 = [lvl.emb for lvl in enc_out2.patches.levels]
        if enc_out3 is not None:
            z3 = [lvl.emb for lvl in enc_out3.patches.levels]
            non_emptys = [(l1.non_empty, l2.non_empty, l3.non_empty)
                         for l1, l2, l3 in zip(enc_out1.patches.levels, enc_out2.patches.levels,
                                               enc_out3.patches.levels)]
        else:
            z3 = None
            non_emptys = [(l1.non_empty, l2.non_empty, None)
                         for l1, l2 in zip(enc_out1.patches.levels, enc_out2.patches.levels)]

        loss_dict = calc_enc_loss_multiscale(z1, z2, self.global_step, self.image_size, z3=z3,
                                             deltas=deltas, target=target, non_emptys=non_emptys,
                                             loss_weights=lw)

        if self.mep_model is not None:
            mep_target = self.ema_encoder(img2) if self.ema_encoder is not None else enc_out2
            emb_pred, masks = self.mep_model(enc_out2)
            total_mep_loss = 0.0
            for lev, (emb, mask) in enumerate(zip(emb_pred, masks)):
                mep_target_lvl = mep_target.patches[lev].emb.detach()
                mep_loss = (emb[~mask] - mep_target_lvl[~mask]).square().mean() \
                    if (~mask).any() else torch.tensor(0.0, device=emb.device)
                total_mep_loss = total_mep_loss + mep_loss
            total_mep_loss = total_mep_loss / enc_out2.patches.num_levels
            loss_dict['mep'] = total_mep_loss
            loss_dict['loss'] = loss_dict['loss'] + lw.get('lambda_mep', 1.0) * total_mep_loss

        bs = img1.shape[0]
        log = {'{}/loss'.format(stage): loss_dict['loss']}
        for k in ('sim', 'sigreg', 'fact', 'mep', 'anchor'):
            if k in loss_dict:
                v = loss_dict[k]
                log['{}/{}'.format(stage, k)] = v if torch.is_tensor(v) else torch.tensor(float(v))
        self.log_dict(log, prog_bar=True, on_step=(stage == 'train'), on_epoch=True,
                     batch_size=bs, sync_dist=(stage != 'train'))
        return loss_dict['loss']

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, 'val')

    def configure_optimizers(self):
        params = list(self.encoder.parameters())
        if self.mep_model is not None:
            params += list(self.mep_model.parameters())
        lr = float(self.tr.get('lr', 1e-3))
        optimizer = torch.optim.AdamW(params, lr=lr)

        epochs = int(self.tr.get('epochs', 5))
        warmup_epochs = int(self.tr.get('lr_warmup_epochs', 3))
        tail_epochs = int(self.tr.get('lr_tail_epochs', 10))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            if epoch < epochs - tail_epochs:
                return 1.0
            progress = (epoch - (epochs - tail_epochs)) / max(1, tail_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}
