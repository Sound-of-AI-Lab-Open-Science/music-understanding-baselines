"""Build and train MusicBERT masked-LM pre-training with PyTorch Lightning.

This module is both a standalone trainer and the thing
``baselines/models/registry.py`` calls: its ``build(config)`` turns a config
dict into (datamodule, LightningModule), which is what makes a recipe in
``baselines/configs/`` and a hand-written config produce the same objects.
Normally you go through the scaffold instead:

    ../run_pretrain.sh musicbert [--smoke]
    python baselines/train.py --model musicbert --config baselines/configs/musicbert.yaml ...

Selected fields can be overridden from the CLI (handy for smoke runs)::

    --data-dir --max-steps --init-from --batch-size --max-octuples --devices

The YAML schema has four sections: ``data``, ``model``, ``optim``, ``trainer``
(plus a top-level ``seed``).  See the configs for documented defaults.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import lightning as L  # noqa: E402
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import CSVLogger  # noqa: E402

from src.data.mlm_datamodule import MusicBertMlmDataModule  # noqa: E402
from src.tasks.mlm_pretrain import MusicBertMlmLitModule  # noqa: E402


class AsciiLogCallback(Callback):
    """Print step/loss/acc/lr in plain ASCII (Rich progress bar breaks on GBK
    Windows consoles)."""

    def __init__(self, every_n: int = 10):
        self.every_n = every_n

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.every_n != 0 and step != 1:
            return
        m = trainer.callback_metrics
        def g(k):
            v = m.get(k)
            try:
                return float(v)
            except Exception:
                return float("nan")
        print("[step {:>6d}] loss={:.4f} acc={:.4f} lr={:.3e}".format(
            step, g("train/loss_step"), g("train/acc_step"), g("lr")), flush=True)


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
    model_cfg = dict(config.get("model", {}))
    optim_cfg = dict(config.get("optim", {}))

    data_cfg["data_dir"] = _resolve(data_cfg["data_dir"])
    dm = MusicBertMlmDataModule(**data_cfg)

    lit = MusicBertMlmLitModule(
        arch=model_cfg.get("arch", "base"),
        model_overrides=model_cfg.get("model_overrides"),
        init_from=_resolve(optim_cfg["init_from"]) if optim_cfg.get("init_from") else None,
        init_strict=optim_cfg.get("init_strict", True),
        peak_lr=float(optim_cfg.get("peak_lr", 5e-4)),
        warmup_updates=int(optim_cfg.get("warmup_updates", 25000)),
        total_num_update=int(optim_cfg.get("total_num_update", 125000)),
        end_learning_rate=float(optim_cfg.get("end_learning_rate", 0.0)),
        power=float(optim_cfg.get("power", 1.0)),
        adam_betas=tuple(optim_cfg.get("adam_betas", (0.9, 0.98))),
        adam_eps=float(optim_cfg.get("adam_eps", 1e-6)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
    )
    return dm, lit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-octuples", type=int, default=None)
    ap.add_argument("--devices", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    with open(_resolve(args.config), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.data_dir:
        config.setdefault("data", {})["data_dir"] = args.data_dir
    if args.batch_size:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.max_octuples:
        config.setdefault("data", {})["max_octuples"] = args.max_octuples
    if args.init_from:
        config.setdefault("optim", {})["init_from"] = args.init_from
    if args.max_steps is not None:
        config.setdefault("trainer", {})["max_steps"] = args.max_steps
    if args.devices is not None:
        config.setdefault("trainer", {})["devices"] = int(args.devices)

    seed = int(config.get("seed", 1))
    L.seed_everything(seed, workers=True)

    dm, lit = build(config)

    tr = dict(config.get("trainer", {}))
    out_dir = _resolve(args.out_dir or tr.pop(
        "default_root_dir",
        os.path.join(_pkg_root_default("RUN_ROOT", os.path.join(_PROJECT_ROOT, "runs")),
                     "mlm")))
    os.makedirs(out_dir, exist_ok=True)

    logger = CSVLogger(save_dir=out_dir, name="", flush_logs_every_n_steps=10)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        AsciiLogCallback(every_n=int(tr.get("log_every_n_steps", 10))),
        ModelCheckpoint(dirpath=os.path.join(out_dir, "checkpoints"),
                        save_last=True, save_top_k=1, monitor="val/loss",
                        mode="min", filename="musicbert-mlm-{step:06d}"),
    ]

    trainer = L.Trainer(
        max_steps=int(tr.get("max_steps", 500)),
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

    trainer.fit(lit, datamodule=dm)

    final = os.path.join(out_dir, "musicbert_mlm_final.pt")
    lit.model.save_pretrained(final)
    print("[train] saved final model to {}".format(final))


if __name__ == "__main__":
    main()
