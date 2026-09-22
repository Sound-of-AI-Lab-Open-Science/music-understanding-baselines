"""Piano-roll tokenization cache for the midi_rae arms (``midi_rae_enc``/``midi_rae_dec``).

Wraps the tempo-normalized 32nd-note piano-roll rasterization from
``pretrain/midi_rae/scripts/midi_to_pianoroll.py`` (now removed as a standalone
script -- its logic lives here) into the same ``build_*_cache(files, out_dir,
workers=..., force=...) -> dict`` contract ``octuple_cache.py``/``remi_cache.py``
are documented to follow (see ``baselines/README.md``), with the same
``index.json`` convention so ``cache_dir_for``'s fingerprinting stays
consistent for this arm.

On-disk layout, matching ``midi_rae.data.ShardedAnchorDataset`` /
``ShardedTripletDataset``'s expectations directly (no adapter needed):

    <out_dir>/
        train_shard00000.pt   train_shard00000.idx.json   ...
        val_shard00000.pt     val_shard00000.idx.json     ...
        index.json            {"meta": {...}}

Each ``*_shard*.pt`` is a Python list of ``uint8`` numpy arrays, shape
``(128, T)`` (T varies per piece); the sidecar ``.idx.json`` holds only
``{"count": N}`` so the lazy dataset can build its index without loading the
tensor file.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import warnings
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


def fingerprint(files: Sequence[str], extra: str = "") -> str:
    """Deterministic ``<N>-<hash>`` tag for exactly this file list.

    Same idea as the octuple/remi caches (hash the sorted file list, not the
    file contents -- a rename or a moved corpus is a cache miss, not a hidden
    bug): the fingerprint covers path + size + mtime per file plus ``extra``
    (the cache kind), so two models that consumed different codecs never share
    a directory by accident.
    """
    h = hashlib.sha1(extra.encode("utf-8"))
    for f in sorted(files):
        try:
            st = os.stat(f)
            h.update("{}\t{}\t{}\n".format(f, st.st_size, int(st.st_mtime)).encode("utf-8"))
        except OSError:
            h.update("{}\tMISSING\n".format(f).encode("utf-8"))
    return "{}-{}".format(len(files), h.hexdigest()[:16])


def _midi_to_roll(path: str, steps_per_beat: int = 8, max_len: int = 4096,
                  max_duration_sec: float = 1800.0) -> Optional[np.ndarray]:
    """Tempo-normalized piano roll: columns are 1/steps_per_beat-note subdivisions
    (steps_per_beat=8 -> 32nd notes), so a given musical duration (in beats)
    stays the same pixel width regardless of the piece's tempo."""
    import pretty_midi

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = pretty_midi.PrettyMIDI(path)
    end_time = pm.get_end_time()
    if end_time <= 0 or end_time > max_duration_sec:
        return None
    tempi = pm.get_tempo_changes()[1]
    try:
        tempo = float(tempi[0]) if len(tempi) else pm.estimate_tempo()
    except Exception:
        tempo = 120.0
    if not (20 <= tempo <= 300):
        tempo = 120.0
    fs = (tempo / 60.0) * steps_per_beat
    roll = pm.get_piano_roll(fs=fs)
    roll = (roll > 0).astype(np.uint8)
    if roll.shape[1] > max_len:
        roll = roll[:, :max_len]
    if roll.shape[1] < 8:
        return None
    return roll


def _worker(args) -> Optional[np.ndarray]:
    path, steps_per_beat, max_len = args
    try:
        return _midi_to_roll(path, steps_per_beat=steps_per_beat, max_len=max_len)
    except Exception:
        return None


def _write_shard(records: list, out_dir: str, split: str, shard_idx: int) -> None:
    path = os.path.join(out_dir, "{}_shard{:05d}.pt".format(split, shard_idx))
    torch.save(records, path)
    with open(path.replace(".pt", ".idx.json"), "w", encoding="utf-8") as f:
        json.dump({"count": len(records)}, f)


def build_pianoroll_cache(files: Sequence[str], out_dir: str, *,
                          workers: int = 8, force: bool = False,
                          steps_per_beat: int = 8, max_len: int = 4096,
                          shard_size: int = 2000, val_frac: float = 0.02,
                          seed: int = 42) -> Dict:
    """Rasterize ``files`` into sharded piano-roll tensors under ``out_dir``.

    Idempotent like the other caches: if ``index.json`` already exists and
    ``force`` is False, the existing cache is trusted and reused (the directory
    name already fingerprints the exact file list).
    """
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.json")
    if os.path.isfile(index_path) and not force:
        with open(index_path, "r", encoding="utf-8") as f:
            meta = json.load(f)["meta"]
        print("[pianoroll] cache already built at {}: {} sequences".format(
            out_dir, meta.get("num_sequences")), flush=True)
        return {"out_dir": os.path.abspath(out_dir), **meta}

    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_frac)
    splits = {"val": shuffled[:n_val], "train": shuffled[n_val:]}

    stats = {"train_total": len(splits["train"]), "val_total": len(splits["val"]),
             "train_ok": 0, "val_ok": 0, "failed": 0}
    failures: List[str] = []

    for split, paths in splits.items():
        buf: list = []
        shard_idx = 0
        job_args = [(p, steps_per_beat, max_len) for p in paths]
        pool_workers = max(1, workers)
        if pool_workers > 1 and len(job_args) > 1:
            with Pool(pool_workers) as pool:
                results = pool.imap(_worker, job_args, chunksize=64)
                for path, roll in zip(paths, results):
                    if roll is None:
                        stats["failed"] += 1
                        failures.append(path)
                        continue
                    buf.append(roll)
                    stats["{}_ok".format(split)] += 1
                    if len(buf) >= shard_size:
                        _write_shard(buf, out_dir, split, shard_idx)
                        shard_idx += 1
                        buf = []
        else:
            for path, args_ in zip(paths, job_args):
                roll = _worker(args_)
                if roll is None:
                    stats["failed"] += 1
                    failures.append(path)
                    continue
                buf.append(roll)
                stats["{}_ok".format(split)] += 1
                if len(buf) >= shard_size:
                    _write_shard(buf, out_dir, split, shard_idx)
                    shard_idx += 1
                    buf = []
        if buf:
            _write_shard(buf, out_dir, split, shard_idx)
            shard_idx += 1
        # A split with zero successes has no shard files at all, and
        # ShardedAnchorDataset asserts on that -- surface it here, loudly, at
        # cache-build time rather than as an opaque AssertionError deep in a
        # DataLoader worker.
        if stats["{}_ok".format(split)] == 0:
            print("[pianoroll] WARNING: 0 usable {} files (of {})".format(
                split, len(paths)), flush=True)
        print("[pianoroll] [{}] wrote {} shards, {} records".format(
            split, shard_idx, stats["{}_ok".format(split)]), flush=True)

    if failures:
        with open(os.path.join(out_dir, "failures.txt"), "w", encoding="utf-8") as f:
            for p in failures:
                f.write("unreadable_or_too_short\t{}\n".format(p))

    meta = {
        "kind": "pianoroll",
        "num_sequences": stats["train_ok"] + stats["val_ok"],
        "num_train": stats["train_ok"], "num_val": stats["val_ok"],
        "num_skipped": stats["failed"],
        "steps_per_beat": steps_per_beat, "max_len": max_len,
        "shard_size": shard_size, "val_frac": val_frac, "seed": seed,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta}, f, indent=2, sort_keys=True)
    print("[pianoroll] DONE {}".format(stats), flush=True)
    return {"out_dir": os.path.abspath(out_dir), **meta}
