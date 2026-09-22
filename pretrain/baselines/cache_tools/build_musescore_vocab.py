"""Derive a MuseTok event dictionary from the MuseScore REMI+ shard caches.

    python baselines/cache_tools/build_musescore_vocab.py \
        --cache-root "$CACHE_ROOT" \
        --out "$VOCAB_ROOT/dictionary_musescore.pkl" \
        [--require-shards 20] [--workers 16]

Why this exists rather than ``data.rebuild_vocab: true``
--------------------------------------------------------
Upstream's ``events2dictionary`` does ``os.listdir(event_path)`` and
``pickle.load``s every entry.  That is non-recursive, so pointed at a merged
union cache it tries to unpickle the ``s00/``..``s19/`` shard symlinks and dies;
pointed at one shard it sees only that shard.  Neither covers the corpus.

Why the released dictionary is not usable here
----------------------------------------------
``third_party/MuseTok/data/dictionary.pkl`` is a 167-event vocabulary derived
from a piano corpus with 12 time signatures.  MuseScore-big is multi-instrument
and far more metrically diverse, so ``convert_event``'s ``event2idx['<name>_<value>']`` raises
``KeyError`` on the first piece carrying anything outside it -- which is how the
200-step smoke died on ``Time_Signature_1/8``.

The output keeps upstream's own convention exactly -- union with
``build_full_vocab()``, then ``sorted(set(...), key=lambda x: (not isinstance(x, int), x))``
-- so the file is a drop-in for ``REMIEventDataset.read_vocab``.  Note that the
sort means ids SHIFT relative to the released dictionary: a checkpoint trained
against this vocabulary is NOT interchangeable with the public MuseTok weights,
and anything assuming ``n_token == 168`` must not be pointed at it.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# The package's one path configuration (published_code/pkgpaths.py). It reads
# paths.yaml and the same environment variables env.sh exports, so a default
# printed by --help is the path this package will actually use.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pkgpaths import PATHS  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _scan_one(path):
    """Worker: -> (events set, n_pieces, pitch_low, pitch_high, worst_bar_span)."""
    try:
        with open(path, "rb") as f:
            pos, events = pickle.load(f)
    except Exception:
        return None
    evs = set()
    low, high = 128, 0
    # per-bar pitch span: upstream's transpose loop retries until the shift fits
    # [min_pitch, max_pitch]; a bar wider than that window never fits and the
    # loop spins forever, so the span is worth measuring while we are here.
    bar_low, bar_high, worst = 128, 0, 0
    for e in events:
        name, value = e["name"], e["value"]
        evs.add("{}_{}".format(name, value))
        if name == "Note_Pitch":
            v = int(value)
            low = min(low, v); high = max(high, v)
            bar_low = min(bar_low, v); bar_high = max(bar_high, v)
        elif name == "Bar":
            if bar_high >= bar_low:
                worst = max(worst, bar_high - bar_low)
            bar_low, bar_high = 128, 0
    if bar_high >= bar_low:
        worst = max(worst, bar_high - bar_low)
    return evs, 1, low, high, worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", default=str(PATHS.cache_root))
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-shards", type=int, default=0,
                    help="fail unless at least this many finished REMI+ shard "
                         "caches are found (use 20 for the real run)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-per-shard", type=int, default=0,
                    help="sample N pieces per shard instead of all (diagnostics only; "
                         "NEVER use for a vocabulary a real run depends on)")
    ap.add_argument("--released-vocab",
                    default=str(PATHS.musetok_repo / "data" / "dictionary.pkl"))
    args = ap.parse_args()

    shards = sorted(d for d in glob.glob(os.path.join(args.cache_root, "remi", "*"))
                    if os.path.isfile(os.path.join(d, "index.json")))
    print("[vocab] finished REMI+ shard caches: {}".format(len(shards)), flush=True)
    for d in shards:
        print("[vocab]   {}".format(os.path.basename(d)), flush=True)
    if args.require_shards and len(shards) < args.require_shards:
        raise SystemExit("[vocab] FATAL: need {} finished shard caches, found {}. "
                         "A vocabulary built on a subset can still miss events and "
                         "KeyError mid-run.".format(args.require_shards, len(shards)))
    if not shards:
        raise SystemExit("[vocab] FATAL: no finished REMI+ shard caches found")
    if args.limit_per_shard:
        print("[vocab] WARNING sampling {} pieces/shard -- diagnostics only, this "
              "vocabulary is NOT safe to train against".format(args.limit_per_shard),
              flush=True)

    files = []
    for d in shards:
        ev = os.path.join(d, "events")
        names = sorted(os.listdir(ev))
        if args.limit_per_shard:
            names = names[:args.limit_per_shard]
        files.extend(os.path.join(ev, n) for n in names)
    print("[vocab] scanning {} event pickles with {} workers".format(
        len(files), args.workers), flush=True)

    all_events = set()
    n_pieces = 0
    low, high, worst_span = 128, 0, 0
    t0 = time.time()
    with Pool(processes=max(1, args.workers)) as pool:
        for i, r in enumerate(pool.imap_unordered(_scan_one, files, chunksize=256), 1):
            if r is None:
                continue
            evs, n, lo, hi, w = r
            all_events |= evs
            n_pieces += n
            low = min(low, lo); high = max(high, hi); worst_span = max(worst_span, w)
            if i % 100000 == 0:
                print("[vocab] {}/{} {:.0f}s".format(i, len(files), time.time() - t0),
                      flush=True)

    from baselines.data.remi_cache import ensure_musetok_on_path
    ensure_musetok_on_path()
    from data_processing.events2words import build_full_vocab

    full = build_full_vocab(add_velocity=False)
    union = sorted(set(list(all_events) + list(full)),
                   key=lambda x: (not isinstance(x, int), x))
    event2word = {k: i for i, k in enumerate(union)}
    word2event = {i: k for i, k in enumerate(union)}

    with open(args.released_vocab, "rb") as f:
        released = set(pickle.load(f)[0])
    novel = sorted(e for e in all_events if e not in released)
    by_name = collections.Counter(e.rsplit("_", 1)[0] for e in novel)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump((event2word, word2event), f)
    os.replace(tmp, args.out)

    print("[vocab] pieces scanned:            {}".format(n_pieces), flush=True)
    print("[vocab] distinct events in corpus: {}".format(len(all_events)), flush=True)
    print("[vocab] released dictionary:       {} events (n_token 168)".format(len(released)))
    print("[vocab] VOCAB SIZE: {} events -> n_token {} (pad appended by read_vocab)"
          .format(len(union), len(union) + 1), flush=True)
    print("[vocab] events absent from the released dictionary: {}".format(len(novel)))
    print("[vocab]   by event name: {}".format(dict(by_name.most_common())), flush=True)
    print("[vocab] Note_Pitch range in corpus: {}..{}".format(low, high), flush=True)
    print("[vocab] widest per-bar pitch span:  {} semitones".format(worst_span), flush=True)
    print("[vocab] wrote {}".format(args.out), flush=True)
    novel_path = os.path.splitext(args.out)[0] + "_novel_events.txt"
    with open(novel_path, "w", encoding="utf-8") as f:
        f.write("".join(e + "\n" for e in novel))
    print("[vocab] full novel-event list -> {}".format(novel_path), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
