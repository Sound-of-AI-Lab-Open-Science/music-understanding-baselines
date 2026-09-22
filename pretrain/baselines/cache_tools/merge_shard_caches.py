"""Union N per-shard tokenization caches into ONE cache directory, no re-tokenizing.

Why this exists
---------------
``baselines/train.py`` names a cache directory by ``fingerprint(files)`` -- a hash
over the EXACT file list.  Tokenizing MuseScore is therefore run as one SLURM
array task per corpus shard (``<corpus>/0`` .. ``<corpus>/19``), which leaves 20
independent cache dirs per codec, and adding a ``--filter-csv`` afterwards would
change the file list, change the hash, and force a full re-tokenize (~6 h Octuple,
days for REMI+).  Both are unacceptable, so this script merges the shard caches
purely at the index-metadata level:

  * OCTUPLE: ``<out>/tokens_<global>.npy`` are RELATIVE SYMLINKS to the source
    shards (token data is never copied); each sequence's ``shard`` field is
    renumbered to the new global index.
  * REMI+:   ``<out>/events/s<NN>`` is one relative symlink per SOURCE cache
    (20 entries, not 1.7 M -- a single GPFS directory of 1.7 M files is a
    performance disaster), and piece names become ``s<NN>/<piece>.pkl``, which
    upstream ``REMIEventDataset`` resolves with ``os.path.join(data_dir, p)``.

An OPTIONAL keep-list (``--keep-ids`` / ``--split-file``) subsets the union
post hoc; it only drops index entries, so applying a filter after tokenization
costs seconds instead of days.

Duplicate ids (``--dedup-ids``, DEFAULT ``prefer-mxl``)
------------------------------------------------------
The MuseScore corpus is NOT one file per id: 123,715 of the 1,565,579 ids (7.9%)
carry BOTH ``<id>.mid`` and ``<id>.mxl.mid`` in the same shard directory, and the
two are byte-identical, so both tokenizers encode ~7.3% of the corpus twice.  By
default this script collapses each id to ONE item, keeping the ``.mxl``-derived
one (else the lexicographically first name).  That takes the effective corpus
from 1,689,294 items down to 1,565,579.  Pass ``--dedup-ids none`` to keep every
item (Octuple still drops exact md5 collisions -- the md5 in the Octuple index is
a hash of the FILE BYTES, so byte-identical twins collide there anyway).

Song-level split (``--split-dir``)
----------------------------------
``--split-dir <dir>`` reads ``ids_{train,val,test}.txt`` written by
``pretrain/scripts/make_musescore_split.py``, restricts the union to ``train UNION val``
(TEST IDS NEVER ENTER THE CACHE -- a held-out song must not be reachable from
the training corpus at all) and, for the REMI codec, writes
``<dir>/pieces_train.txt`` / ``<dir>/pieces_val.txt``: the MERGED piece names
(``s<NN>/<id>.mxl.pkl``) of the ids that actually survived tokenization, sorted.
Those two files are what ``data.train_pieces_file`` / ``data.val_pieces_file``
in the MuseTok config point at -- only the merge knows the ``s<NN>`` prefix,
because it is assigned here.  The Octuple path needs no such file: it consumes
``ids_{train,val}.txt`` directly via ``data.train_ids_file`` /
``data.val_ids_file``.  ``ids_test.txt`` is only read, never rewritten.

Usage::

    python baselines/cache_tools/merge_shard_caches.py --codec octuple \
        --cache-root "$CACHE_ROOT" \
        --out "$CACHE_ROOT/union_octuple" \
        [--keep-ids ids_train.txt | --split-file splits/.../pool_train.json] [--dry-run]

The resulting directory is consumed by ``baselines/train.py --cache-prebuilt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# The package's one path configuration (published_code/pkgpaths.py). It reads
# paths.yaml and the same environment variables env.sh exports, so a default
# printed by --help is the path this package will actually use.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pkgpaths import PATHS  # noqa: E402

INDEX_NAME = "index.json"
EVENTS_SUBDIR = "events"

#: meta keys that MUST agree across every shard cache (they define the codec).
_OCTUPLE_INVARIANTS = ("kind", "vocab_size", "tokens_per_note", "pad_id", "bos_id",
                       "eos_id", "dtype")


# ---------------------------------------------------------------------------
# ids
# ---------------------------------------------------------------------------
def id_of(filename: str) -> str:
    """``100000.mxl.mid`` -> ``100000``; strips EVERY extension (the ``match: id``).

    Also handles ``100000.mxl.pkl`` (REMI piece) and ``100000.mid``, and a piece
    name carrying a subdirectory (``s03/100000.mxl.pkl`` -> ``100000``).
    """
    stem = os.path.basename(filename)
    while True:
        root, ext = os.path.splitext(stem)
        if not ext or not root:
            return stem
        stem = root


def load_keep_ids(path: str) -> Set[str]:
    """Keep-list from a JSON list, a pool file, a newline txt, or a CSV."""
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if isinstance(blob, list):
            items = blob
        elif isinstance(blob, dict) and "songs" in blob:      # pool_*.json
            items = [s["song_id"] if isinstance(s, dict) else s for s in blob["songs"]]
        elif isinstance(blob, dict):
            items = blob.get("ids", [])
        else:
            items = []
        return {id_of(str(x)) for x in items}
    keep: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip().split(",")[0].strip()
            if tok and tok.lower() not in ("id", "song_id", "name", "filename"):
                keep.add(id_of(tok))
    return keep


def _variant_rank(name: str) -> Tuple[int, str]:
    """Sort key picking the winner among same-id items: ``.mxl`` first, then name.

    ``100000.mxl.mid`` beats ``100000.mid``; ``100000.mxl.pkl`` beats ``100000.pkl``.
    Deterministic and independent of shard iteration order.
    """
    base = os.path.basename(name)
    root = os.path.splitext(base)[0]
    return (0 if root.endswith(".mxl") else 1, base)


def _keep_one_per_id(names: Sequence[str], sids: Sequence[str]) -> Set[int]:
    """Positions to KEEP: one per id, chosen by :func:`_variant_rank`."""
    best: Dict[str, Tuple[Tuple[int, str], int]] = {}
    for pos, (name, sid) in enumerate(zip(names, sids)):
        key = (_variant_rank(name), pos)
        if sid not in best or key < best[sid]:
            best[sid] = key
    return {pos for _, pos in best.values()}


def load_split_dir(split_dir: str) -> Dict[str, Set[str]]:
    """``ids_{train,val,test}.txt`` from a make_musescore_split.py output dir.

    ``ids_test.txt`` is READ ONLY so its ids can be excluded; it is never rewritten.
    """
    out: Dict[str, Set[str]] = {}
    for split in ("train", "val", "test"):
        path = os.path.join(split_dir, "ids_{}.txt".format(split))
        if not os.path.isfile(path):
            raise SystemExit("[merge] --split-dir {}: missing {}".format(
                split_dir, os.path.basename(path)))
        out[split] = load_keep_ids(path)
    return out


def _write_lines(path: str, lines: Sequence[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(x + "\n" for x in lines))
    os.replace(tmp, path)


def union_fingerprint(sources: Sequence[str], keep_identity: Optional[str],
                      n_items: int, dedup_ids: str = "prefer-mxl") -> str:
    """Deterministic id for "this union of these caches under this keep-list".

    Shaped like ``octuple_cache.fingerprint`` so ``cache_is_current`` accepts it
    and never tries to rebuild a merged cache in place.
    """
    h = hashlib.sha1()
    h.update(b"union-v1")
    for name in sorted(os.path.basename(os.path.normpath(s)) for s in sources):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
    h.update((keep_identity or "-").encode("utf-8"))
    h.update(("dedup=" + str(dedup_ids)).encode("utf-8"))
    return "{}-{}".format(n_items, h.hexdigest()[:12])


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def discover_shard_caches(cache_root: str, codec: str, out_dir: str) -> List[str]:
    """Every subdirectory of ``<cache_root>/<codec>/``, sorted by name."""
    base = os.path.join(cache_root, codec)
    if not os.path.isdir(base):
        raise FileNotFoundError("no such cache root: {}".format(base))
    out_abs = os.path.abspath(out_dir)
    dirs = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if os.path.isdir(path) and os.path.abspath(path) != out_abs:
            dirs.append(path)
    return dirs


def _count_key(codec: str) -> str:
    return "num_sequences" if codec == "octuple" else "num_pieces"


def load_shard_indexes(dirs: Sequence[str], codec: str) -> Tuple[List[Tuple[str, dict]],
                                                                List[Tuple[str, str]]]:
    """Return ``(usable, skipped)``; a still-running or failed job is SKIPPED loudly."""
    usable: List[Tuple[str, dict]] = []
    skipped: List[Tuple[str, str]] = []
    key = _count_key(codec)
    items_key = "sequences" if codec == "octuple" else "pieces"
    for d in dirs:
        p = os.path.join(d, INDEX_NAME)
        if not os.path.isfile(p):
            skipped.append((d, "no index.json (tokenization still running?)"))
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError) as exc:
            skipped.append((d, "unreadable index.json: {}".format(exc)))
            continue
        meta = blob.get("meta") or {}
        if meta.get("kind") not in (None, codec):
            skipped.append((d, "wrong codec: kind={}".format(meta.get("kind"))))
            continue
        if int(meta.get(key, 0)) <= 0 or not blob.get(items_key):
            skipped.append((d, "empty cache ({}=0)".format(key)))
            continue
        usable.append((d, blob))
    return usable, skipped


# ---------------------------------------------------------------------------
# atomic-ish output directory handling
# ---------------------------------------------------------------------------
def _prepare_out_dir(out_dir: str) -> str:
    """Return a fresh staging dir next to ``out_dir`` (swapped in at the end)."""
    staging = os.path.abspath(out_dir) + ".partial"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    return staging


def _commit_out_dir(staging: str, out_dir: str) -> None:
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir):
        old = out_dir + ".old"
        if os.path.exists(old):
            shutil.rmtree(old)
        os.rename(out_dir, old)
        os.rename(staging, out_dir)
        shutil.rmtree(old)
    else:
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        os.rename(staging, out_dir)


def write_index(out_dir: str, blob: dict) -> None:
    """Never leave a half-written index.json behind."""
    final = os.path.join(out_dir, INDEX_NAME)
    tmp = final + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    os.replace(tmp, final)


def _relative_symlink(target: str, link: str) -> None:
    rel = os.path.relpath(os.path.abspath(target), os.path.dirname(os.path.abspath(link)))
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(rel, link)


def _merge_common_meta(metas: Sequence[dict]) -> dict:
    reasons: Dict[str, int] = {}
    for m in metas:
        for k, v in (m.get("skip_reasons") or {}).items():
            reasons[k] = reasons.get(k, 0) + int(v)
    return {
        "num_input_files": sum(int(m.get("num_input_files", 0)) for m in metas),
        "num_skipped": sum(int(m.get("num_skipped", 0)) for m in metas),
        "skip_reasons": dict(sorted(reasons.items())),
    }


# ---------------------------------------------------------------------------
# octuple
# ---------------------------------------------------------------------------
def merge_octuple(sources: Sequence[Tuple[str, dict]], out_dir: str,
                  keep: Optional[Set[str]], keep_identity: Optional[str],
                  dry_run: bool, dedup_ids: str = "prefer-mxl") -> dict:
    first_meta = sources[0][1]["meta"]
    for d, blob in sources[1:]:
        for k in _OCTUPLE_INVARIANTS:
            if blob["meta"].get(k) != first_meta.get(k):
                raise SystemExit(
                    "[merge] ABORT: incompatible caches -- {} is {!r} in {} but {!r} "
                    "in {}. These were tokenized with different settings; merging "
                    "them would silently corrupt training.".format(
                        k, blob["meta"].get(k), d, first_meta.get(k), sources[0][0]))

    shard_files: List[str] = []
    symlinks: List[Tuple[str, str]] = []          # (target, link basename)
    cand: List[dict] = []                         # records, shard already global
    cand_src: List[int] = []
    cand_sid: List[str] = []
    cand_name: List[str] = []
    per_source_total: List[Tuple[str, int]] = []
    seen_ids: Set[str] = set()
    n_dropped_keep = 0
    g = 0

    for si, (d, blob) in enumerate(sources):
        meta = blob["meta"]
        local_to_global = {}
        for local, fname in enumerate(meta["shard_files"]):
            gname = "tokens_{:05d}.npy".format(g)
            local_to_global[local] = g
            shard_files.append(gname)
            symlinks.append((os.path.join(d, fname), gname))
            g += 1
        for rec in blob["sequences"]:
            name = rec.get("name") or rec.get("path") or rec["md5"]
            sid = id_of(name)
            seen_ids.add(sid)
            if keep is not None and sid not in keep:
                n_dropped_keep += 1
                continue
            new_rec = dict(rec)
            new_rec["shard"] = local_to_global[int(rec["shard"])]
            cand.append(new_rec)
            cand_src.append(si)
            cand_sid.append(sid)
            cand_name.append(name)
        per_source_total.append((d, len(blob["sequences"])))

    # (1) Same id, several files (<id>.mid AND <id>.mxl.mid -- 7.9% of MuseScore,
    # byte-identical): collapse to one, keeping the .mxl-derived name.  Done
    # BEFORE the md5 dedup, so the survivor is the PREFERRED variant rather than
    # whichever copy the encoder happened to reach first.
    n_id_dropped = 0
    if dedup_ids != "none":
        keep_pos = _keep_one_per_id(cand_name, cand_sid)
        if len(keep_pos) != len(cand):
            n_id_dropped = len(cand) - len(keep_pos)
            sel = sorted(keep_pos)
            cand = [cand[i] for i in sel]
            cand_src = [cand_src[i] for i in sel]

    # (2) Exact content duplicates, first-wins.  The md5 in an Octuple index is a
    # hash of the FILE BYTES (pretrain/scripts/preprocess_lmd.py::_md5_of_path only reuses
    # the filename stem when it is a 32-hex Lakh md5, never the case for
    # MuseScore), so byte-identical twins collide here even across shards.
    sequences: List[dict] = []
    by_md5: Dict[str, int] = {}
    kept_per_source: Dict[int, int] = {}
    n_md5_dropped = 0
    for rec, si in zip(cand, cand_src):
        if rec["md5"] in by_md5:
            n_md5_dropped += 1
            continue
        by_md5[rec["md5"]] = len(sequences)
        sequences.append(rec)
        kept_per_source[si] = kept_per_source.get(si, 0) + 1

    per_shard = [(d, n, kept_per_source.get(si, 0))
                 for si, (d, n) in enumerate(per_source_total)]

    metas = [b["meta"] for _, b in sources]
    meta_out = {
        "kind": "octuple",
        "fingerprint": union_fingerprint([d for d, _ in sources], keep_identity,
                                         len(sequences), dedup_ids),
        "vocab_size": first_meta["vocab_size"],
        "tokens_per_note": first_meta["tokens_per_note"],
        "pad_id": first_meta["pad_id"],
        "bos_id": first_meta["bos_id"],
        "eos_id": first_meta["eos_id"],
        "num_shards": len(shard_files),
        "shard_files": shard_files,
        "num_sequences": len(sequences),
        "dtype": first_meta.get("dtype", "int32"),
        "union_of": [os.path.abspath(d) for d, _ in sources],
        "keep_list": keep_identity,
        "dedup_ids": dedup_ids,
        **_merge_common_meta(metas),
    }
    stats = {"per_shard": per_shard,
             "total_before": sum(n for _, n in per_source_total),
             "total_after": len(sequences), "md5_dropped": n_md5_dropped,
             "id_dropped": n_id_dropped, "keep_dropped": n_dropped_keep,
             "seen_ids": seen_ids,
             "kept_names": [r.get("name") or r["md5"] for r in sequences]}

    if dry_run:
        return {"meta": meta_out, "stats": stats}

    staging = _prepare_out_dir(out_dir)
    for target, gname in symlinks:
        _relative_symlink(target, os.path.join(staging, gname))
    write_index(staging, {"meta": meta_out, "sequences": sequences, "by_md5": by_md5})
    _commit_out_dir(staging, out_dir)
    return {"meta": meta_out, "stats": stats}


# ---------------------------------------------------------------------------
# remi
# ---------------------------------------------------------------------------
def merge_remi(sources: Sequence[Tuple[str, dict]], out_dir: str,
               keep: Optional[Set[str]], keep_identity: Optional[str],
               dry_run: bool, dedup_ids: str = "prefer-mxl") -> dict:
    pieces: List[str] = []
    piece_sids: List[str] = []
    per_shard: List[Tuple[str, int, int]] = []
    symlinks: List[Tuple[str, str]] = []
    seen_ids: Set[str] = set()
    seen_names: Set[str] = set()
    n_dropped_keep = 0
    n_dup = 0

    for i, (d, blob) in enumerate(sources):
        prefix = "s{:02d}".format(i)
        symlinks.append((os.path.join(d, EVENTS_SUBDIR), prefix))
        kept_here = 0
        for p in blob["pieces"]:
            sid = id_of(p)
            seen_ids.add(sid)
            if keep is not None and sid not in keep:
                n_dropped_keep += 1
                continue
            name = "{}/{}".format(prefix, p)
            if name in seen_names:
                n_dup += 1
                continue
            seen_names.add(name)
            pieces.append(name)
            piece_sids.append(sid)
            kept_here += 1
        per_shard.append((d, len(blob["pieces"]), kept_here))

    # REMI has no content hash, so <id>.pkl and <id>.mxl.pkl are simply two
    # distinct piece names for the same (byte-identical) song: dedup by id.
    n_id_dropped = 0
    if dedup_ids != "none":
        keep_pos = _keep_one_per_id(pieces, piece_sids)
        if len(keep_pos) != len(pieces):
            n_id_dropped = len(pieces) - len(keep_pos)
            pieces = [p for i, p in enumerate(pieces) if i in keep_pos]

    metas = [b["meta"] for _, b in sources]
    meta_out = {
        "kind": "remi",
        "fingerprint": union_fingerprint([d for d, _ in sources], keep_identity,
                                         len(pieces), dedup_ids),
        "num_pieces": len(pieces),
        "union_of": [os.path.abspath(d) for d, _ in sources],
        "keep_list": keep_identity,
        "dedup_ids": dedup_ids,
        **_merge_common_meta(metas),
    }
    stats = {"per_shard": per_shard, "total_before": sum(x[1] for x in per_shard),
             "total_after": len(pieces), "md5_dropped": n_dup,
             "id_dropped": n_id_dropped, "keep_dropped": n_dropped_keep,
             "seen_ids": seen_ids, "kept_names": list(pieces)}

    if dry_run:
        return {"meta": meta_out, "stats": stats, "pieces": pieces}

    staging = _prepare_out_dir(out_dir)
    events = os.path.join(staging, EVENTS_SUBDIR)
    os.makedirs(events)
    for target, prefix in symlinks:
        _relative_symlink(target, os.path.join(events, prefix))
    write_index(staging, {"meta": meta_out, "pieces": pieces})
    _commit_out_dir(staging, out_dir)
    return {"meta": meta_out, "stats": stats, "pieces": pieces}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def merge(codec: str, out_dir: str, cache_root: Optional[str] = None,
          shard_caches: Optional[Sequence[str]] = None,
          keep_ids: Optional[str] = None, split_file: Optional[str] = None,
          dry_run: bool = False, quiet: bool = False,
          dedup_ids: str = "prefer-mxl",
          split_dir: Optional[str] = None) -> dict:
    """Merge shard caches; returns ``{"meta":..., "stats":...}``."""
    def say(msg):
        if not quiet:
            print(msg, flush=True)

    if shard_caches:
        dirs = [os.path.abspath(d) for d in shard_caches]
    else:
        if not cache_root:
            raise SystemExit("[merge] need --cache-root or --shard-cache")
        dirs = discover_shard_caches(cache_root, codec, out_dir)
        say("[merge] auto-discovered {} candidate dirs under {}/{}".format(
            len(dirs), cache_root, codec))

    usable, skipped = load_shard_indexes(dirs, codec)
    for d, why in skipped:
        say("[merge] SKIPPED {}: {}".format(d, why))
    if skipped:
        say("[merge] {} of {} candidate cache dirs were skipped (listed above) -- "
            "if a tokenization job is still running, re-run this merge when it "
            "finishes.".format(len(skipped), len(dirs)))
    if not usable:
        raise SystemExit("[merge] ABORT: no usable shard caches found")

    keep = None
    keep_identity = None
    if keep_ids and split_file:
        raise SystemExit("[merge] pass only one of --keep-ids / --split-file")
    if keep_ids:
        keep = load_keep_ids(keep_ids)
        keep_identity = os.path.abspath(keep_ids)
    elif split_file:
        keep = load_keep_ids(split_file)
        keep_identity = os.path.abspath(split_file)

    splits = None
    if split_dir:
        splits = load_split_dir(split_dir)
        trainval = splits["train"] | splits["val"]
        say("[merge] split-dir {}: train {} / val {} / test {} ids; the test ids "
            "are EXCLUDED from the cache entirely".format(
                os.path.abspath(split_dir), len(splits["train"]),
                len(splits["val"]), len(splits["test"])))
        keep = trainval if keep is None else (keep & trainval)
        keep_identity = "{}|{}".format(keep_identity or "-",
                                       os.path.abspath(split_dir))
    if keep is not None:
        say("[merge] keep-list {}: {} ids".format(keep_identity, len(keep)))

    if dedup_ids not in ("none", "prefer-mxl"):
        raise SystemExit("[merge] --dedup-ids must be none|prefer-mxl")
    fn = merge_octuple if codec == "octuple" else merge_remi
    result = fn(usable, out_dir, keep, keep_identity, dry_run, dedup_ids)
    st = result["stats"]

    say("[merge] per-shard (source, items, kept):")
    for d, n, k in st["per_shard"]:
        say("[merge]   {:<60s} {:>8d} -> {:>8d}".format(os.path.basename(d), n, k))
    say("[merge] total before keep-list/dedup: {}".format(st["total_before"]))
    if st["keep_dropped"]:
        say("[merge] dropped by keep-list:                 {}".format(
            st["keep_dropped"]))
    # Printed ALWAYS, not only when non-zero: at full MuseScore scale these two
    # numbers should together account for ~123715 duplicate ids (7.9% of the
    # corpus carries both <id>.mid and <id>.mxl.mid, byte-identical).  A number
    # far off that is a red flag about the tokenization inputs, not a detail.
    say("[merge] dropped as md5 duplicates (first kept): {}".format(
        st["md5_dropped"]))
    say("[merge] dropped as same-id variants ({:>10s}): {}".format(
        dedup_ids, st["id_dropped"]))
    say("[merge] total after  keep-list/dedup: {}".format(st["total_after"]))
    if keep is not None:
        missing = keep - st["seen_ids"]
        extra = st["seen_ids"] - keep
        say("[merge] keep-list ids with no cached item: {}".format(len(missing)))
        say("[merge] cached ids not in the keep-list:  {}".format(len(extra)))
    if splits is not None:
        buckets = {"train": [], "val": []}
        for name in st["kept_names"]:
            sid = id_of(name)
            if sid in splits["train"]:
                buckets["train"].append(name)
            elif sid in splits["val"]:
                buckets["val"].append(name)
        for split in ("train", "val"):
            buckets[split].sort()
            have = {id_of(n) for n in buckets[split]}
            say("[merge] {}: {} cached items covering {} ids; {} ids have NO "
                "cached item (tokenization failure, or not in these shards)".format(
                    split, len(buckets[split]), len(have),
                    len(splits[split] - have)))
        leaked = sum(1 for n in st["kept_names"] if id_of(n) in splits["test"])
        say("[merge] held-out test ids present in the union: {} (MUST be 0)".format(
            leaked))
        if leaked:
            raise SystemExit("[merge] ABORT: {} held-out test songs leaked into "
                             "the union cache".format(leaked))
        if codec == "remi":
            written = []
            for split in ("train", "val"):
                path = os.path.join(split_dir, "pieces_{}.txt".format(split))
                if not dry_run:
                    _write_lines(path, buckets[split])
                written.append(path)
            say("[merge] {} {}".format(
                "DRY-RUN, would write" if dry_run else "wrote", ", ".join(written)))
            say("[merge] point data.train_pieces_file / data.val_pieces_file at "
                "those two files")
        else:
            say("[merge] octuple needs no piece list: point data.train_ids_file / "
                "data.val_ids_file at {}/ids_{{train,val}}.txt".format(
                    os.path.abspath(split_dir)))
        result["split_counts"] = {k: len(v) for k, v in buckets.items()}

    say("[merge] {}{}".format("DRY-RUN, nothing written; would write "
                             if dry_run else "wrote ", os.path.abspath(out_dir)))
    result["stats"].pop("seen_ids", None)
    result["stats"].pop("kept_names", None)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec", required=True, choices=["octuple", "remi"])
    ap.add_argument("--cache-root", default=str(PATHS.cache_root))
    ap.add_argument("--out", required=True, help="union cache directory to create")
    ap.add_argument("--shard-cache", action="append", default=None, dest="shard_caches",
                    help="explicit source cache dir (repeatable); "
                         "default = every dir under <cache-root>/<codec>/")
    ap.add_argument("--keep-ids", default=None,
                    help="keep-list: newline txt, CSV (first column), or JSON list")
    ap.add_argument("--split-file", default=None,
                    help="pool_*.json split file; keeps only its song_ids")
    ap.add_argument("--dedup-ids", default="prefer-mxl",
                    choices=["none", "prefer-mxl"],
                    help="collapse ids that have several files (<id>.mid AND "
                         "<id>.mxl.mid, byte-identical, 7.9%% of MuseScore) to one "
                         "item, keeping the .mxl-derived name; default prefer-mxl "
                         "takes the corpus from 1689294 items to 1565579")
    ap.add_argument("--split-dir", default=None,
                    help="a pretrain/scripts/make_musescore_split.py output dir: "
                         "restricts "
                         "the union to train+val ids (test ids never enter the "
                         "cache) and, for remi, writes pieces_{train,val}.txt "
                         "there for data.{train,val}_pieces_file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    merge(args.codec, args.out, cache_root=args.cache_root,
          shard_caches=args.shard_caches, keep_ids=args.keep_ids,
          split_file=args.split_file, dry_run=args.dry_run,
          dedup_ids=args.dedup_ids, split_dir=args.split_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
