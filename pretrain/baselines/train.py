"""Single entry point for the three symbolic-music baselines.

    python baselines/train.py --model jepa      --config baselines/configs/jepa_paper.yaml --midi-dir <dir>
    python baselines/train.py --model musicbert --config baselines/configs/musicbert.yaml  --midi-dir <dir>
    python baselines/train.py --model musetok   --config baselines/configs/musetok.yaml    --midi-dir <dir>

Normally you do not call this directly: ``run_pretrain.sh <arm> [--smoke]``
picks the config, the interpreter and the union cache for an arm and submits it.

Add ``--filter-csv <csv>`` to restrict the corpus, ``--max-steps N`` to cap
training, and ``--smoke`` to run the tiny end-to-end proof (a handful of files,
a handful of steps, CPU or GPU) that the pipeline works: ``--smoke`` merges the
config's own ``smoke:`` block over the recipe, so the smoke of an arm is defined
next to the arm and cannot drift from it.

Every string in the config is passed through ``os.path.expandvars``, so a YAML
may write ``${SPLIT_ROOT}/main/ids_train.txt`` and stay relocatable. An
undefined variable is left verbatim -- and will fail as a missing file, loudly.

Which interpreter: the ``jepa-music`` environment for jepa/musicbert, the
``musetok`` environment for musetok -- see baselines/README.md.

The run does, in order: discover MIDI -> apply the filter CSV -> build (or reuse)
the on-disk tokenization cache -> build the model -> fit -> write a checkpoint.
Tokenization is cached per (model-codec, exact file list), so the second run of
any model on the same corpus starts training immediately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from baselines.data.midi_index import FilterConfig, select_files  # noqa: E402
from baselines.models import registry  # noqa: E402

_BASELINES_DIR = os.path.dirname(os.path.abspath(__file__))


def _pkg_default(var: str, fallback_name: str) -> str:
    """Where a writable tree lives: $WORK_ROOT's, not the checkout's.

    ``env.sh`` exports ``$CACHE_ROOT`` / ``$RUN_ROOT`` and ``pkgpaths.py``
    resolves the same values, so the normal path is the environment variable.
    Importing ``pkgpaths`` fills them in for a bare ``python baselines/train.py``
    with no env.sh sourced. Only if both fail does this fall back to a directory
    inside the checkout, which ``.gitignore`` covers.
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
    return value or os.path.join(_BASELINES_DIR, fallback_name)


DEFAULT_CACHE_ROOT = _pkg_default("CACHE_ROOT", "cache")
DEFAULT_RUN_ROOT = _pkg_default("RUN_ROOT", "runs")


def expand_env(node: Any) -> Any:
    """Expand ``${VAR}`` in every string of a loaded config, recursively.

    This is what lets one recipe be run from any checkout: the configs name
    ``${SPLIT_ROOT}``, ``${CACHE_ROOT}`` and ``${VOCAB_ROOT}``, which env.sh
    exports and pkgpaths.py resolves identically. ``os.path.expandvars`` leaves
    an undefined variable untouched rather than blanking it, so a forgotten
    ``source env.sh`` surfaces as "no such file: ${SPLIT_ROOT}/..." instead of a
    silent fallback to the wrong corpus.
    """
    if isinstance(node, str):
        return os.path.expandvars(node)
    if isinstance(node, dict):
        return {k: expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [expand_env(v) for v in node]
    return node


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(
            out.get(k), dict) else v
    return out


def _filter_config(config: dict, csv_path: Optional[str]) -> FilterConfig:
    """Build the CSV filter from ``data.filter`` in the YAML plus ``--filter-csv``.

    The section is POPPED: ``data`` is splatted straight into the upstream
    DataModule constructors, which reject any key they do not declare.
    """
    raw = dict(config.setdefault("data", {}).pop("filter", None) or {})
    raw.pop("path", None)
    return FilterConfig(path=csv_path, **raw)


class StepLogger:
    """Plain-ASCII per-step line: no Rich, no unicode, greppable in SLURM logs."""

    def __init__(self, callback_base, every_n: int, metrics):
        self.every_n = max(1, every_n)
        self.metrics = list(metrics)
        self._t0 = time.time()

        logger = self

        class _Cb(callback_base):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                step = trainer.global_step
                if step % logger.every_n and step != 1:
                    return
                m = trainer.callback_metrics

                def g(key):
                    try:
                        return float(m[key])
                    except (KeyError, TypeError, ValueError):
                        return float("nan")

                body = " ".join("{}={:.4f}".format(k.split("/")[-1].replace("_step", ""),
                                                   g(k)) for k in logger.metrics)
                print("[step {:>6d}] {} elapsed={:.0f}s".format(
                    step, body, time.time() - logger._t0), flush=True)

        self.callback = _Cb()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as f:
        config = expand_env(yaml.safe_load(f) or {})

    smoke_cfg = config.pop("smoke", {}) or {}
    limit = args.limit
    if args.smoke:
        limit = limit or int(smoke_cfg.pop("limit_files", 20))
        config = deep_merge(config, smoke_cfg)
        print("[train] SMOKE mode: {} files, overrides {}".format(limit, smoke_cfg),
              flush=True)
    if args.max_steps is not None:
        config.setdefault("trainer", {})["max_steps"] = args.max_steps
    if args.devices is not None:
        config.setdefault("trainer", {})["devices"] = args.devices
    if args.accelerator is not None:
        config.setdefault("trainer", {})["accelerator"] = args.accelerator

    if args.cache_prebuilt:
        # A union cache merged from the per-shard tokenization jobs
        # (baselines/cache_tools/merge_shard_caches.py).  Discovery, filtering and
        # tokenization are ALL skipped: --midi-dir is never walked, and a
        # post-hoc keep-list lives in the merged index instead of changing the
        # file list (which would change the cache hash and force a re-tokenize).
        cache_dir = os.path.abspath(args.cache_prebuilt)
        cache = registry.load_prebuilt_cache(args.model, cache_dir, config=config)
        config.setdefault("data", {}).pop("filter", None)
        files = []
        n_files = int(cache.get("num_sequences", cache.get("num_pieces", 0)))
        print("[train] prebuilt cache {} ({} items)".format(cache_dir, n_files),
              flush=True)
    else:
        files = select_files(args.midi_dir, _filter_config(config, args.filter_csv),
                             limit=limit)
        if args.filter_only:
            # Dry run of discovery + filtering only: validating a filter CSV against
            # a 1.7M-file corpus must not require tokenizing it first.
            return {"num_files": len(files), "skipped_tokenizing": True}

        cache_root = args.cache_dir or DEFAULT_CACHE_ROOT
        cache_dir = registry.cache_dir_for(args.model, cache_root, files)
        cache = registry.prepare_cache(args.model, files,
                                       os.path.abspath(args.midi_dir),
                                       cache_dir, workers=args.workers,
                                       force=args.force_tokenize, config=config,
                                       resolve_vocab=not args.tokenize_only)
        n_files = len(files)
    if args.tokenize_only:
        print("[train] CACHE_DIR {}".format(cache_dir), flush=True)
        return {"cache": cache_dir, "skipped_training": True}

    # Seed BEFORE the model is constructed, so weight init is reproducible.
    # Both Lightning packages delegate to the same lightning_fabric RNG seeding,
    # so it does not matter which one does it.
    from lightning_fabric.utilities.seed import seed_everything
    seed_everything(int(config.get("seed", 1)), workers=True)

    built = registry.build(args.model, config, cache)
    L = built.lightning

    tr = dict(config.get("trainer", {}))
    out_dir = os.path.abspath(args.out_dir or os.path.join(
        DEFAULT_RUN_ROOT, "{}_smoke".format(args.model) if args.smoke else args.model))
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    cb, lg = built.callbacks_mod, built.loggers_mod
    every_n = int(tr.get("log_every_n_steps", 10))
    callbacks = [
        cb.LearningRateMonitor(logging_interval="step"),
        StepLogger(cb.Callback, every_n, built.step_metrics).callback,
        # monitor=None: a periodic "latest" checkpoint that is always written,
        # so a run killed by the SLURM walltime is resumable even if validation
        # never ran.  A best-val checkpoint is added separately below.
        cb.ModelCheckpoint(dirpath=ckpt_dir, monitor=None, save_last=True,
                           save_top_k=1,
                           every_n_train_steps=int(tr.get("ckpt_every_n_steps", 1000)),
                           filename="{}-periodic-{{step:06d}}".format(args.model)),
    ]
    if built.monitor:
        callbacks.append(cb.ModelCheckpoint(
            dirpath=ckpt_dir, monitor=built.monitor, mode="min", save_top_k=1,
            save_last=False, filename="{}-best-{{step:06d}}".format(args.model)))

    # An INTEGER val_check_interval means "every N batches" and Lightning, by
    # default, resets that counter at every epoch boundary -- so it refuses any
    # value larger than one epoch:
    #   ValueError: `val_check_interval` (100) must be less than or equal to the
    #   number of the training batches (6).
    # Every recipe here states it in steps (100 in a smoke block, 2000 in a
    # production one) because that is what it means on a corpus whose epoch is
    # far longer than the interval. On a SMALL corpus -- which is exactly what
    # --smoke and a pilot run use -- the epoch is shorter than the interval and
    # the run dies at fit() before its first step, having already paid for
    # tokenization. check_val_every_n_epoch=None is Lightning's own remedy (the
    # error message names it): it counts val_check_interval against total
    # training batches instead of per-epoch ones, which is the recipes' intent
    # at any corpus size. A FLOAT interval is a fraction of an epoch and must
    # keep the per-epoch counter, so it is left alone.
    _vci = tr.get("val_check_interval", 1.0)
    _check_val_every_n_epoch = None if isinstance(_vci, int) else 1

    trainer = L.Trainer(
        max_steps=int(tr.get("max_steps", -1)),
        max_epochs=tr.get("max_epochs", -1),
        check_val_every_n_epoch=_check_val_every_n_epoch,
        accelerator=tr.get("accelerator", "auto"),
        devices=tr.get("devices", 1),
        precision=tr.get("precision", "32-true"),
        accumulate_grad_batches=int(tr.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(tr.get("gradient_clip_val", 0.0)) or None,
        log_every_n_steps=every_n,
        val_check_interval=tr.get("val_check_interval", 1.0),
        limit_val_batches=tr.get("limit_val_batches", 1.0),
        num_sanity_val_steps=int(tr.get("num_sanity_val_steps", 0)),
        enable_progress_bar=bool(tr.get("enable_progress_bar", False)),
        logger=lg.CSVLogger(save_dir=out_dir, name="", flush_logs_every_n_steps=10),
        callbacks=callbacks,
    )

    resume = os.path.join(ckpt_dir, "last.ckpt")
    ckpt_path = resume if (args.resume and os.path.isfile(resume)) else None
    t0 = time.time()
    trainer.fit(built.module, datamodule=built.datamodule, ckpt_path=ckpt_path)
    wall = time.time() - t0

    # Always write a final checkpoint: a smoke run can finish before any periodic
    # or best-val trigger fires, and "did a checkpoint appear" is what the test asserts.
    final_ckpt = os.path.join(ckpt_dir, "final.ckpt")
    trainer.save_checkpoint(final_ckpt)

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()
               if hasattr(v, "item") or isinstance(v, (int, float))}
    summary = {
        "model": args.model, "out_dir": out_dir, "cache": cache_dir,
        "steps": int(trainer.global_step), "wall_seconds": round(wall, 1),
        "checkpoint": final_ckpt, "metrics": metrics,
        "num_files": n_files, "num_skipped_tokenizing": cache.get("num_skipped", 0),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[train] DONE steps={} wall={:.1f}s ckpt={}".format(
        summary["steps"], wall, final_ckpt), flush=True)
    print("[train] final metrics: {}".format(
        json.dumps(metrics, sort_keys=True)), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=registry.MODELS)
    ap.add_argument("--config", required=True)
    ap.add_argument("--midi-dir", default=None,
                    help="directory tree of .mid/.midi files (searched recursively); "
                         "required unless --cache-prebuilt is given")
    ap.add_argument("--cache-prebuilt", default=None,
                    help="train directly on a prebuilt/union cache directory "
                         "(built by baselines/cache_tools/merge_shard_caches.py); "
                         "skips MIDI discovery, filtering and tokenization")
    ap.add_argument("--filter-csv", default=None,
                    help="CSV restricting the corpus; column names come from "
                         "data.filter in the YAML")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset + few steps, to prove the pipeline end to end")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N files after filtering")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cache-dir", default=None, help="root for tokenization caches")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="processes used for tokenization only")
    ap.add_argument("--force-tokenize", action="store_true",
                    help="rebuild the cache even if it is current")
    ap.add_argument("--tokenize-only", action="store_true")
    ap.add_argument("--filter-only", action="store_true",
                    help="stop after discovery + filtering; print the counts only")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out-dir>/checkpoints/last.ckpt when present")
    ap.add_argument("--devices", default=None)
    ap.add_argument("--accelerator", default=None, choices=["cpu", "gpu", "auto"])
    args = ap.parse_args()
    if args.cache_prebuilt:
        if args.tokenize_only or args.filter_only:
            ap.error("--cache-prebuilt trains on an already-tokenized cache; it "
                     "cannot be combined with --tokenize-only/--filter-only")
        if args.filter_csv:
            ap.error("--cache-prebuilt cannot be combined with --filter-csv: apply "
                     "the keep-list when merging (merge_shard_caches.py --keep-ids), "
                     "not at train time")
    elif not args.midi_dir:
        ap.error("--midi-dir is required unless --cache-prebuilt is given")
    if args.devices is not None:
        args.devices = int(args.devices)
    run(args)


if __name__ == "__main__":
    main()
