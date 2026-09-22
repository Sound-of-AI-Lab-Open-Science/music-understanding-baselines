"""LightningModule for MusicBERT masked-LM pre-training.

Wraps :class:`src.models.musicbert.MusicBert` and reproduces the official fairseq
training recipe extracted from ``train_mask.sh`` and the checkpoint ``cfg``:

    optimizer = Adam(betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    lr_scheduler = polynomial_decay(power=1.0): linear warmup to peak_lr over
        ``warmup_updates`` then linear decay to ``end_learning_rate`` at
        ``total_num_update``.
    peak_lr = 5e-4, warmup_updates = 25000, total_num_update = 125000.

Logs ``train/loss``, ``train/acc`` (top-1 over masked targets), ``val/loss``,
``val/acc`` and the current learning rate.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch

try:
    import lightning as L
except Exception:  # pragma: no cover
    import pytorch_lightning as L  # type: ignore

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.musicbert import MusicBert, MusicBertConfig, config_for_arch  # noqa: E402

IGNORE_INDEX = -100


class MusicBertMlmLitModule(L.LightningModule):
    def __init__(
        self,
        arch: str = "base",
        model_overrides: Optional[dict] = None,
        # optimization (official defaults)
        peak_lr: float = 5e-4,
        warmup_updates: int = 25000,
        total_num_update: int = 125000,
        end_learning_rate: float = 0.0,
        power: float = 1.0,
        adam_betas=(0.9, 0.98),
        adam_eps: float = 1e-6,
        weight_decay: float = 0.01,
        # optional warm start from a converted checkpoint
        init_from: Optional[str] = None,
        init_strict: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        config = config_for_arch(arch, **(model_overrides or {}))
        self.config = config
        self.model = MusicBert(config)

        if init_from:
            self._load_weights(init_from, strict=init_strict)

    # -- init from converted checkpoint -------------------------------------
    def _load_weights(self, path: str, strict: bool = True):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            # Asymmetric on purpose. `unexpected` is legitimate here: the checkpoint
            # carries lm_head/upsampling that this model does not have. `missing` is
            # not -- every backbone parameter must receive weights, or the run trains
            # or scores a partly random model. On 2026-08-09 exactly that happened
            # elsewhere: a config-less checkpoint left all six layers' attention at
            # random init and the probe scored it anyway.
            raise RuntimeError(
                'incomplete backbone load: %d parameters left at random init '
                '(%d unexpected keys ignored)' % (len(missing), len(unexpected)))

        if strict and unexpected:
            # `missing` is handled unconditionally above; only extra keys remain
            # optional, and they are the benign case (lm_head/upsampling the model
            # does not carry). Leaving `missing` in this clause as well made the
            # unconditional guard look redundant while the strict-gated one did the
            # real work -- and `strict` defaults to False.
            raise RuntimeError(
                "init_from load not clean: unexpected={}".format(unexpected))
        print("[mlm] initialized weights from {} (missing={}, unexpected={})".format(
            path, len(missing), len(unexpected)))

    # -- forward / steps ----------------------------------------------------
    def forward(self, input_ids, labels=None):
        return self.model(input_ids, labels=labels, ignore_index=IGNORE_INDEX)

    @staticmethod
    def _accuracy(logits, labels):
        with torch.no_grad():
            mask = labels != IGNORE_INDEX
            if mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device)
            pred = logits.argmax(-1)
            return (pred[mask] == labels[mask]).float().mean()

    def training_step(self, batch, batch_idx):
        out = self.model(batch["input_ids"], labels=batch["labels"], ignore_index=IGNORE_INDEX)
        loss = out["loss"]
        acc = self._accuracy(out["logits"], batch["labels"])
        bs = batch["input_ids"].shape[0]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("lr", self._current_lr(), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch["input_ids"], labels=batch["labels"], ignore_index=IGNORE_INDEX)
        acc = self._accuracy(out["logits"], batch["labels"])
        bs = batch["input_ids"].shape[0]
        self.log("val/loss", out["loss"], prog_bar=True, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, on_epoch=True, batch_size=bs, sync_dist=True)
        return out["loss"]

    def _current_lr(self):
        opts = self.optimizers()
        if isinstance(opts, (list, tuple)):
            opts = opts[0]
        try:
            return opts.param_groups[0]["lr"]
        except Exception:
            return float("nan")

    # -- keep the datamodule's per-epoch masking seed in sync ---------------
    def on_train_epoch_start(self):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "set_epoch"):
            dm.set_epoch(self.current_epoch)

    # -- optimizer + polynomial-decay schedule ------------------------------
    def configure_optimizers(self):
        # no weight decay on biases / LayerNorm (standard practice)
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(
            groups, lr=self.hparams.peak_lr,
            betas=tuple(self.hparams.adam_betas), eps=self.hparams.adam_eps,
        )

        warmup = max(1, int(self.hparams.warmup_updates))
        total = max(warmup + 1, int(self.hparams.total_num_update))
        end_ratio = (self.hparams.end_learning_rate / self.hparams.peak_lr
                     if self.hparams.peak_lr > 0 else 0.0)
        power = float(self.hparams.power)

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            pct = (step - warmup) / max(1, (total - warmup))
            pct = min(pct, 1.0)
            return max(end_ratio, (1.0 - pct) ** power * (1.0 - end_ratio) + end_ratio)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
