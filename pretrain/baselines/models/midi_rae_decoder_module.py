"""LightningModule port of midi-rae's decoder training loop
(``pretrain/midi_rae/midi_rae/train_dec.py::train`` + ``train_step``, now
removed as a standalone script -- this is the integrated replacement).

Trains a ``SwinDecoder`` to reconstruct the piano-roll crop from a FROZEN
encoder's embeddings (``encoder_ckpt`` in the config, matching upstream's
``config_swin_full_backup.yaml: encoder_ckpt: checkpoints/SwinEncoder_..._best.pt``
-- here it points at the Lightning checkpoint the ``midi_rae_enc`` arm writes,
since the frozen encoder is now this pipeline's own output rather than a
separately-run script's).

Scope note: upstream's decoder loop also supports an optional
``emb_mask_ratio`` "robustness" training path (randomly masking encoder
embeddings before decoding) and a learnable-mask-token variant. Neither is set
in ``config_swin_full_backup.yaml`` (``emb_mask_ratio`` is absent -> 0.0, a
no-op in upstream too), so it is not ported here; a config that wants it will
need that path added first -- this is a scope reduction, not a silent drop of
something the shipped recipe exercises.
"""

from __future__ import annotations

from typing import Optional

import lightning as L
import torch

from baselines.models.midi_rae_lib.losses import calc_dec_loss
from baselines.models.midi_rae_lib.swin import SwinDecoder, SwinEncoder


def _load_frozen_encoder(encoder: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    """Load encoder weights from either a Lightning checkpoint (``state_dict``
    with an ``encoder.`` prefix -- what ``midi_rae_enc`` writes) or a bare
    ``state_dict`` (e.g. an upstream-style checkpoint)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = {k[len('encoder.'):]: v for k, v in ckpt['state_dict'].items()
              if k.startswith('encoder.')}
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
    else:
        sd = ckpt
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    print("[midi_rae_dec] loaded encoder from {} (missing={}, unexpected={})".format(
        ckpt_path, len(missing), len(unexpected)), flush=True)
    return encoder


class MidiRaeDecoderModule(L.LightningModule):
    """Swin decoder trained to reconstruct piano rolls from a frozen encoder."""

    def __init__(self, model: Optional[dict] = None, training: Optional[dict] = None,
                 image_size: int = 128, encoder_ckpt: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        m = dict(model or {})
        tr = dict(training or {})
        self.image_size = int(image_size)
        self.tr = tr

        depths = tuple(m.get('depths', (2, 2, 2, 6, 2, 1)))
        num_heads = tuple(m.get('num_heads', (2, 2, 2, 4, 8, 16)))
        embed_dim = int(m.get('embed_dim', 8))
        patch_h, patch_w = int(m.get('patch_h', 4)), int(m.get('patch_w', 4))
        window_size = int(m.get('window_size', 4))

        self.encoder = SwinEncoder(
            img_height=image_size, img_width=image_size, patch_h=patch_h, patch_w=patch_w,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, mlp_ratio=float(m.get('mlp_ratio', 4.0)),
            drop_path_rate=float(m.get('drop_path_rate', 0.1)))
        if encoder_ckpt:
            _load_frozen_encoder(self.encoder, encoder_ckpt)
        for p in self.encoder.parameters(): p.requires_grad = False
        self.encoder.eval()

        self.decoder = SwinDecoder(
            img_height=image_size, img_width=image_size, patch_h=patch_h, patch_w=patch_w,
            embed_dim=embed_dim, depths=m.get('dec_depths', depths),
            num_heads=m.get('dec_num_heads', num_heads), window_size=window_size,
            mlp_ratio=float(m.get('mlp_ratio', 4.0)))

    def train(self, mode: bool = True):
        # The encoder is FROZEN: keep it in eval() regardless of the module's
        # own train()/eval() state, so BatchNorm/Dropout-style behaviour (none
        # here, but future-proof) and autograd never touch it.
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, img: torch.Tensor):
        with torch.no_grad():
            enc_out = self.encoder(img)
        return self.decoder(enc_out)

    def _step(self, batch, stage: str):
        img = batch['img']
        note_weights = batch.get('note_weights')
        with torch.no_grad():
            enc_out = self.encoder(img)
        loss_dict = calc_dec_loss(self.decoder, enc_out, img,
                                  pos_weight=float(self.tr.get('pos_weight', 1.0)),
                                  note_weights=note_weights)
        bs = img.shape[0]
        self.log_dict({
            '{}/loss'.format(stage): loss_dict['dec'],
            '{}/bce'.format(stage): loss_dict['bce'],
            '{}/mse'.format(stage): loss_dict['mse'],
        }, prog_bar=True, on_step=(stage == 'train'), on_epoch=True,
            batch_size=bs, sync_dist=(stage != 'train'))
        return loss_dict['dec']

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def configure_optimizers(self):
        lr = float(self.tr.get('dec_lr', 2e-3))
        optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=lr)
        epochs = max(1, int(self.tr.get('dec_epochs', 5)))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=1, epochs=epochs, div_factor=4)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}
