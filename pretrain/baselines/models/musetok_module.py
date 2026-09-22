"""LightningModule wrapping MuseTok's ``TransformerResidualVQ`` tokenizer.

Upstream MuseTok trains with a hand-rolled loop
(``third_party/MuseTok/train_tokenizer.py``); this is the only one of the three
baselines that needed wrapping.  The wrapper is deliberately thin -- the
network and the loss function are upstream's own code, called from the cloned
checkout, and the training recipe below follows upstream's settings rather than
its loop (recorded in THIRD_PARTY_NOTICES.md):

  * loss  = ``recons_ce + beta * commit_loss`` via ``model.compute_loss``
    (``third_party/MuseTok/model/musetok.py``), beta = 1;
  * optim = Adam(lr=max_lr), grad-norm clip 0.5 (set on the Trainer);
  * lr    = linear warmup over ``warmup_steps``, then the closed form of
    ``CosineAnnealingLR(T_max=decay_steps, eta_min=min_lr)`` evaluated at
    ``step - warmup_steps`` -- exactly what upstream's
    ``sched.step(trained_steps - lr_warmup_steps)`` produces.

Requires the ``musetok`` conda env.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import lightning as L
import torch

from baselines.data.remi_cache import ensure_musetok_on_path


class MuseTokLitModule(L.LightningModule):
    """Residual-VQ bar tokenizer trained to reconstruct REMI+ event sequences."""

    def __init__(self, vocab_size: int, model: Optional[dict] = None, *,
                 beta: float = 1.0, max_lr: float = 1e-4, min_lr: float = 5e-6,
                 warmup_steps: int = 200, decay_steps: int = 150000):
        super().__init__()
        self.save_hyperparameters()
        ensure_musetok_on_path()
        from model.musetok import TransformerResidualVQ

        m = dict(model or {})
        self.model = TransformerResidualVQ(
            m.get("enc_n_layer", 12), m.get("enc_n_head", 8),
            m.get("enc_d_model", 512), m.get("enc_d_ff", 2048),
            m.get("dec_n_layer", 12), m.get("dec_n_head", 8),
            m.get("dec_d_model", 512), m.get("dec_d_ff", 2048),
            m.get("d_latent", 128), m.get("d_embed", 512), int(vocab_size),
            m.get("num_quantizers", 16), m.get("codebook_size", 2048),
            rotation_trick=m.get("rotation_trick", True),
            rvq_type=m.get("rvq_type", "SimVQ"),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        # ``TransformerResidualVQ`` expects sequence-first tensors: the encoder
        # input arrives here as (batch, bar, token) and the decoder input as
        # (batch, token), so both are permuted to put time first.
        enc_inp = batch["enc_input"].permute(2, 0, 1)
        dec_inp = batch["dec_input"].permute(1, 0)
        return self.model(enc_inp, dec_inp, batch["bar_pos"],
                          padding_mask=batch["enc_padding_mask"])

    @staticmethod
    def _reconstruction_accuracy(dec_logits, dec_tgt, lengths) -> float:
        """Token accuracy over the unpadded prefix of each sample (upstream formula)."""
        hit = torch.argmax(dec_logits, dim=-1) == dec_tgt
        lens = lengths.tolist()
        correct = sum(int(hit[: max(0, n - 1), i].sum()) for i, n in enumerate(lens))
        return correct / max(1, sum(lens))

    def _step(self, batch, stage: str):
        dec_tgt = batch["dec_target"].permute(1, 0)
        dec_logits, commit_loss = self(batch)
        # beta is zeroed for validation so val/loss is a pure reconstruction
        # number, matching upstream's validate().
        beta = self.hparams.beta if stage == "train" else 0.0
        losses = self.model.compute_loss(commit_loss, beta, dec_logits, dec_tgt)
        acc = self._reconstruction_accuracy(dec_logits, dec_tgt, batch["length"])

        bs = dec_tgt.size(1)
        self.log_dict({
            "{}/loss".format(stage): losses["total_loss"],
            "{}/recons_loss".format(stage): losses["recons_loss"],
            "{}/commit_loss".format(stage): losses["commit_loss"],
            "{}/acc".format(stage): torch.tensor(acc, device=dec_logits.device),
        }, prog_bar=True, on_step=(stage == "train"), on_epoch=True,
            batch_size=bs, sync_dist=(stage != "train"))
        return losses["total_loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.max_lr)
        warmup = max(1, int(self.hparams.warmup_steps))
        decay = max(1, int(self.hparams.decay_steps))
        max_lr, min_lr = float(self.hparams.max_lr), float(self.hparams.min_lr)
        floor = min_lr / max_lr if max_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            t = min(step - warmup, decay)
            cosine = 0.5 * (1.0 + math.cos(math.pi * t / decay))
            return floor + (1.0 - floor) * cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval": "step", "frequency": 1,
            },
        }
