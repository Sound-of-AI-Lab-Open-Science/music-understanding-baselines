"""Song-level train/val/test split over the MuseScore-big-MIDI corpus.

Writes the pool-file split convention -- ``pool_train.json`` / ``pool_val.json`` /
``pool_test.json`` + ``manifest.jsonl`` -- modelled on
the evaluation library's own ``splits/seed<seed>_<ratios>/`` layout.

    python scripts/make_musescore_split.py \
        --corpus "$MIDI_DIR" \
        --out    "$SPLIT_ROOT/main"

The unit is the song: one MuseScore id == one ``<id>.mxl.mid`` file == one row.
There is no chunking here (the humdrum pool files carry a ``chunks`` list per
song because that corpus is chunked; ours is not), so each song carries exactly
one item and ``chunk_size`` is absent.

Determinism, and why it is not a shuffle
----------------------------------------
Assignment is ``md5("<seed>:<song_id>")`` mapped to [0, 1), NOT
``random.shuffle`` of the file list.  Both are reproducible, but the hash is
also *order-independent and stable under corpus growth*: adding or removing
files never moves an unrelated id across the train/test boundary, so a split
made today stays comparable with one made after the corpus is re-exported.
That property is what makes the leakage story defensible.

Optional ``--keep-ids`` restricts the split to a keep-list (the filter CSV, when
one is supplied) without changing any surviving id's assignment --
again a consequence of hashing rather than shuffling.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# The package's one path configuration (published_code/pkgpaths.py). It reads
# paths.yaml and the same environment variables env.sh exports, so a default
# printed by --help is the path this package will actually use.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pkgpaths import PATHS  # noqa: E402

FRACTIONS = {"test": 0.01, "val": 0.01}  # remainder -> train


def id_of(filename: str) -> str:
    """``100000.mxl.mid`` -> ``100000`` (the scaffold's ``match: id``)."""
    stem = os.path.basename(filename)
    while True:
        root, ext = os.path.splitext(stem)
        if not ext:
            return stem
        stem = root


def load_content_hashes(cache_root: str) -> dict:
    """``song_id -> content md5``, read from the Octuple caches' ``index.json``.

    ``scripts/preprocess_lmd.py::_md5_of_path`` falls back to hashing the file
    bytes for any filename that is not already a 32-hex Lakh stem, so the ``md5``
    in an Octuple index IS the MIDI file's content hash.  Reusing it costs one
    JSON read per shard instead of re-hashing 1.7 M files.
    """
    mapping = {}
    n_idx = 0
    for path in sorted(glob.glob(os.path.join(cache_root, "octuple", "*", "index.json"))):
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        for seq in blob.get("sequences", []):
            mapping[id_of(seq["name"])] = seq["md5"]
        n_idx += 1
    print("[split] content hashes: {} ids from {} Octuple shard indexes".format(
        len(mapping), n_idx), flush=True)
    if n_idx and n_idx < 20:
        print("[split] WARNING only {} of 20 Octuple shards are cached. Ids from the "
              "missing shards will fall back to id-keying, which re-opens the "
              "byte-twin leak. Do not use this split for a real run.".format(n_idx),
              flush=True)
    if not n_idx:
        raise SystemExit(
            "[split] --content-hash-from {}: no Octuple index.json found. "
            "Tokenization must finish first.".format(cache_root))
    return mapping


def bucket(song_id: str, seed: int, content_hash: str = None) -> str:
    """Stable assignment, keyed on CONTENT when a content hash is known.

    Keying on the id alone is not enough.  Measured on 15 of the 20 shards:
    1,161,850 files carry only 1,004,390 distinct content hashes, and 1,286 of
    the 10,634 test ids present (12.1%) have a BYTE-IDENTICAL twin sitting in
    the train split under a different MuseScore id.  An id-level split excludes
    the test *id* while leaving its duplicate content in training, which is
    exactly the leak the split exists to prevent.  Hashing the content instead
    sends every byte-identical copy to the same side of the partition.

    ``content_hash`` is None for ids no Octuple cache covers (a tokenization
    failure); those fall back to the id, which is safe because a file that never
    tokenized is not in any training cache either.
    """
    key = content_hash or song_id
    h = hashlib.md5("{}:{}".format(seed, key).encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    if u < FRACTIONS["test"]:
        return "test"
    if u < FRACTIONS["test"] + FRACTIONS["val"]:
        return "val"
    return "train"


def load_keep_ids(path: str) -> set:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        items = blob if isinstance(blob, list) else blob.get("ids", [])
        return {str(x) for x in items}
    keep = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip().split(",")[0].strip()
            if tok and tok.lower() not in ("id", "song_id"):
                keep.add(id_of(tok))
    return keep


def discover(corpus: str):
    """Yield ``(song_id, shard, relpath, n_variants)`` for every id under ``corpus``.

    Sorted at every level so the manifest order is reproducible.

    The corpus is NOT one file per id.  123,715 of the 1,565,579 ids carry BOTH
    ``<id>.mid`` and ``<id>.mxl.mid``, and spot checks show the pairs are
    byte-identical -- the same score exported twice.  Both reduce to the same id
    under ``match: id``, so the song-level unit here is the id and we emit one
    row per id, deterministically preferring the ``.mxl.mid`` variant (the one
    derived from MuseScore-big-MXL, whose ``metadata.jsonl`` is keyed the same
    way).  ``n_variants`` records how many files backed the id, so the duplicate
    rate stays visible downstream instead of silently doubling the corpus.
    """
    for shard in sorted(os.listdir(corpus)):
        shard_dir = os.path.join(corpus, shard)
        if not os.path.isdir(shard_dir):
            continue
        by_id = {}
        for name in sorted(os.listdir(shard_dir)):
            if name.endswith((".mid", ".midi")):
                by_id.setdefault(id_of(name), []).append(name)
        for song_id in sorted(by_id):
            names = by_id[song_id]
            pick = next((n for n in names if n.endswith(".mxl.mid")), names[0])
            yield song_id, shard, "{}/{}".format(shard, pick), len(names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(PATHS.midi_dir))
    ap.add_argument("--out", required=True, help="split directory to create")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-ids", default=None,
                    help="optional keep-list (txt/csv/json); ids absent from it are dropped")
    ap.add_argument("--content-hash-from", default=None,
                    help="root of the Octuple caches; assign splits by FILE CONTENT "
                         "hash rather than id, so byte-identical duplicates under "
                         "different ids cannot straddle the train/test boundary")
    ap.add_argument("--hash-missing", action="store_true",
                    help="md5 the actual MIDI files for ids no Octuple cache covers "
                         "(the ~19k tokenization failures). Without this they fall "
                         "back to id-keying, and any byte-twin among them can "
                         "straddle the split in the REMI+ corpus, which tokenizes a "
                         "different subset than Octuple does.")
    ap.add_argument("--test-fraction", type=float, default=0.01)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    args = ap.parse_args()

    FRACTIONS["test"] = args.test_fraction
    FRACTIONS["val"] = args.val_fraction
    os.makedirs(args.out, exist_ok=True)

    chash = load_content_hashes(args.content_hash_from) if args.content_hash_from else {}
    n_no_chash = [0]
    n_hashed = [0]

    def content_hash_of(song_id, abs_path):
        """Index hash if cached; else hash the file itself when --hash-missing."""
        h = chash.get(song_id)
        if h is not None or not args.hash_missing:
            return h
        try:
            with open(abs_path, "rb") as fh:
                h = hashlib.md5(fh.read()).hexdigest()
            chash[song_id] = h
            n_hashed[0] += 1
            return h
        except OSError:
            return None
    keep = load_keep_ids(args.keep_ids) if args.keep_ids else None
    if keep is not None:
        print("[split] keep-list {}: {} ids".format(args.keep_ids, len(keep)), flush=True)

    pools = {"train": [], "val": [], "test": []}
    seen = set()
    n_dup = n_dropped = 0
    n_extra_variants = [0]
    t0 = time.time()
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, (song_id, shard, relpath, n_variants) in enumerate(discover(args.corpus), 1):
            if song_id in seen:      # ids are globally unique; guard anyway
                n_dup += 1
                continue
            seen.add(song_id)
            if keep is not None and song_id not in keep:
                n_dropped += 1
                continue
            ch = content_hash_of(song_id, os.path.join(args.corpus, relpath))
            if args.content_hash_from and ch is None:
                n_no_chash[0] += 1
            split = bucket(song_id, args.seed, ch)
            rec = {"song_id": song_id, "shard": shard, "relpath": relpath,
                   "midi": os.path.join(args.corpus, relpath), "split": split,
                   "n_variants": n_variants}
            n_extra_variants[0] += n_variants - 1
            mf.write(json.dumps(rec, sort_keys=True) + "\n")
            pools[split].append({"song_id": song_id, "shard": shard,
                                 "relpath": relpath, "midi": rec["midi"],
                                 "n_variants": n_variants})
            if i % 200000 == 0:
                print("[split] {} files, {:.0f}s".format(i, time.time() - t0), flush=True)

    for split, songs in pools.items():
        blob = {"split": split, "seed": args.seed, "corpus": args.corpus,
                "unit": "song", "match": "id",
                "fractions": {"test": args.test_fraction, "val": args.val_fraction,
                              "train": round(1 - args.test_fraction - args.val_fraction, 6)},
                "assignment": ("md5('<seed>:<file content md5>') -> [0,1) threshold"
                               if args.content_hash_from else
                               "md5('<seed>:<song_id>') -> [0,1) threshold"),
                "content_hash_from": args.content_hash_from,
                "keep_ids": os.path.abspath(args.keep_ids) if args.keep_ids else None,
                "song_count": len(songs), "songs": songs}
        with open(os.path.join(args.out, "pool_{}.json".format(split)), "w",
                  encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
        # Flat id list: what merge_shard_caches.py --keep-ids consumes.
        with open(os.path.join(args.out, "ids_{}.txt".format(split)), "w",
                  encoding="utf-8") as f:
            f.write("".join(s["song_id"] + "\n" for s in songs))

    total = sum(len(v) for v in pools.values())
    print("[split] {} songs -> train {} / val {} / test {}  "
          "(cross-shard dup ids {}, dropped-by-keeplist {}, "
          "redundant .mid/.mxl.mid variants collapsed {})"
          .format(total, len(pools["train"]), len(pools["val"]), len(pools["test"]),
                  n_dup, n_dropped, n_extra_variants[0]), flush=True)
    if args.content_hash_from:
        if args.hash_missing:
            print("[split] {} ids hashed directly from disk (not in any Octuple cache)"
                  .format(n_hashed[0]), flush=True)
        print("[split] {} ids had NO content hash and were assigned by id -- MUST be 0 "
              "for a leak-free split".format(n_no_chash[0]), flush=True)
    print("[split] wrote {} in {:.0f}s".format(args.out, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
