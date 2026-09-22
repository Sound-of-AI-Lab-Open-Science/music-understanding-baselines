"""Encode the Lakh MIDI Dataset (lmd_full) into sharded OctupleMIDI token ids.

Deterministic, offline preprocessing.  For every ``*.mid`` under
``data/lmd_full`` we:

  1. parse + encode it with :func:`src.data.octuple.encode_file`
     (the deterministic path -- no training-time augmentation),
  2. serialize the primary window with
     :func:`src.data.octuple.encoding_to_str` (``bar_index_offset=0``, first
     ``SAMPLE_LEN_MAX`` notes, bars clamped to ``< BAR_MAX``),
  3. map tokens -> ids with the fairseq-order vocab (:data:`DEFAULT_VOCAB`).

Output layout (under ``--out``)::

    tokens_00000.npy      flat int32 array (concatenated sequences)
    tokens_00001.npy
    ...
    index.json            { meta, sequences:[{md5,shard,offset,length}], by_md5 }
    failures.txt          one "<reason>\t<relpath>" line per skipped/failed file

Random access: ``sequences[i]`` -> slice ``[offset : offset+length]`` of shard
``tokens_{shard:05d}.npy``.  ``by_md5[md5]`` -> index into ``sequences`` (the
Lakh filename stem *is* the file's md5, which the genre labels are keyed on).

Windows-safe: worker is a module-level function and all Pool usage is under the
``if __name__ == '__main__'`` guard (spawn start method).

Examples::

    python scripts/preprocess_lmd.py --limit 500
    python scripts/preprocess_lmd.py --workers 12 --out data/lmd_octuple
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# make project root importable when run as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data import octuple as oct  # noqa: E402
from src.data.vocab import DEFAULT_VOCAB  # noqa: E402

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


DEFAULT_DATA_DIR = os.path.join(
    _pkg_root_default("DATA_ROOT", os.path.join(_PROJECT_ROOT, "data")), "lmd_full")
DEFAULT_OUT_DIR = os.path.join(
    _pkg_root_default("CACHE_ROOT", os.path.join(_PROJECT_ROOT, "data")), "lmd_octuple")
# Flush a shard once it reaches this many int32 elements (~256 MB per shard).
SHARD_MAX_ELEMS = 64 * 1024 * 1024


def _md5_of_path(path: str) -> str:
    """The Lakh filename stem is the content md5; fall back to hashing if not."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if len(stem) == 32 and all(c in "0123456789abcdef" for c in stem.lower()):
        return stem.lower()
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def encode_one(path):
    """Worker: returns (md5, ids_list_or_None, status_reason, content_hash).

    status_reason is 'ok' on success, else a short reason string.  content_hash
    is the official dedup hash (md5 over (program,pitch) pairs) or None.
    """
    md5 = _md5_of_path(path)
    try:
        encoding = oct.encode_file(path)          # deterministic; raises on reject
        try:
            chash = oct.get_hash(encoding)
        except BaseException:  # noqa: BLE001  match official: hash failure is non-fatal
            chash = None
        tokens = oct.encoding_to_str(encoding).split()
        ids = DEFAULT_VOCAB.encode_tokens(tokens)
        # a valid sequence has at least one real note: 8 <s> + 8 note + 7 </s>
        if len(ids) <= oct.TOKENS_PER_NOTE * 2 - 1:
            return md5, None, "EMPTY", chash
        return md5, ids, "ok", chash
    except oct.MidiEncodeError as exc:
        return md5, None, str(exc).split(":")[0], None  # PARSE / BLANK / ENCODE / TSFILT
    except BaseException as exc:  # noqa: BLE001
        return md5, None, "OTHER({})".format(type(exc).__name__), None


def _list_midi_files(data_dir):
    files = []
    for root, _dirs, names in os.walk(data_dir):
        for n in names:
            low = n.lower()
            if low.endswith(".mid") or low.endswith(".midi"):
                files.append(os.path.join(root, n))
    files.sort()
    return files


class _ShardWriter:
    """Accumulates int32 ids and flushes them into ``tokens_NNNNN.npy`` shards."""

    def __init__(self, out_dir):
        import numpy as np
        self.np = np
        self.out_dir = out_dir
        self.shard_idx = 0
        self.offset = 0            # offset within current shard
        self.buf = []             # list of int32 arrays for current shard
        self.buf_len = 0
        self.sequences = []       # [{md5, shard, offset, length}]
        self.by_md5 = {}
        self.shard_files = []

    def add(self, md5, ids):
        arr = self.np.asarray(ids, dtype=self.np.int32)
        length = int(arr.shape[0])
        self.sequences.append({
            "md5": md5,
            "shard": self.shard_idx,
            "offset": self.offset,
            "length": length,
        })
        self.by_md5[md5] = len(self.sequences) - 1
        self.buf.append(arr)
        self.buf_len += length
        self.offset += length
        if self.buf_len >= SHARD_MAX_ELEMS:
            self._flush()

    def _flush(self):
        if not self.buf:
            return
        data = self.np.concatenate(self.buf)
        fname = "tokens_{:05d}.npy".format(self.shard_idx)
        self.np.save(os.path.join(self.out_dir, fname), data)
        self.shard_files.append(fname)
        self.buf = []
        self.buf_len = 0
        self.shard_idx += 1
        self.offset = 0

    def close(self):
        self._flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N files (for smoke tests)")
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--dedup", action="store_true",
                    help="drop content-duplicate encodings (official dedup hash)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = _list_midi_files(args.data_dir)
    if args.limit is not None:
        files = files[: args.limit]
    total = len(files)
    print("[preprocess] {} MIDI files under {}".format(total, args.data_dir), flush=True)
    if total == 0:
        print("[preprocess] nothing to do", flush=True)
        return

    writer = _ShardWriter(args.out)
    reasons = {}
    n_ok = 0
    failures = []
    seen_hashes = set()
    t0 = time.time()

    from multiprocessing import Pool

    with Pool(processes=args.workers, maxtasksperchild=2000) as pool:
        for i, (md5, ids, reason, chash) in enumerate(
                pool.imap_unordered(encode_one, files, chunksize=args.chunksize), 1):
            if reason == "ok" and args.dedup and chash is not None:
                if chash in seen_hashes:
                    reason = "DUPLICATE"
                else:
                    seen_hashes.add(chash)
            if reason == "ok":
                writer.add(md5, ids)
                n_ok += 1
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
                failures.append((reason, md5))
            if i % 1000 == 0 or i == total:
                el = time.time() - t0
                rate = i / el if el > 0 else 0.0
                print("[preprocess] {}/{}  ok={}  {:.1f} files/s  eta={:.1f} min".format(
                    i, total, n_ok, rate, (total - i) / rate / 60 if rate > 0 else -1),
                    flush=True)

    writer.close()

    index = {
        "meta": {
            "vocab_size": len(DEFAULT_VOCAB),
            "tokens_per_note": oct.TOKENS_PER_NOTE,
            "pad_id": DEFAULT_VOCAB.pad_id,
            "bos_id": DEFAULT_VOCAB.bos_id,
            "eos_id": DEFAULT_VOCAB.eos_id,
            "num_shards": len(writer.shard_files),
            "shard_files": writer.shard_files,
            "num_sequences": len(writer.sequences),
            "dtype": "int32",
        },
        "sequences": writer.sequences,
        "by_md5": writer.by_md5,
    }
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)

    # failures / skips log (ASCII only)
    with open(os.path.join(args.out, "failures.txt"), "w", encoding="utf-8") as f:
        for reason, md5 in failures:
            f.write("{}\t{}\n".format(reason, md5))

    el = time.time() - t0
    rate = total / el if el > 0 else 0.0
    print("[preprocess] DONE ok={} skipped={} shards={} seqs={}".format(
        n_ok, total - n_ok, len(writer.shard_files), len(writer.sequences)), flush=True)
    print("[preprocess] skip reasons: {}".format(dict(sorted(reasons.items()))), flush=True)
    print("[preprocess] {:.1f}s total, {:.1f} files/s".format(el, rate), flush=True)
    if total < 178561 and rate > 0:
        est = 178561 / rate / 3600.0
        print("[preprocess] full-corpus (178,561) estimate at this rate: "
              "{:.2f} h".format(est), flush=True)


if __name__ == "__main__":
    main()
