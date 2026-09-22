"""Cut a tiny, self-consistent copy of a corpus for ``run_smoke.sh --end-to-end``.

    python lib/smoke_corpus.py midi --src <MIDI_DIR> --out <dir> [--files 100] [--shards 2]
    python lib/smoke_corpus.py cipi --src <BENCHMIR_DATA_ROOT/cipi> --out <dir> [--movements 40]

Why a COPY rather than a limit flag
-----------------------------------
Every stage of this pipeline is keyed on the corpus: the tokenization cache is
fingerprinted by the exact file list, the split is keyed on content hashes of
that list, and the union cache and the MuseTok dictionary are derived from the
shard caches. A smoke that pointed the real ``$MIDI_DIR`` at a ``--limit`` would
therefore either take hours or produce artefacts that share a name with the real
run's. A separate small tree, under the smoke's own ``$WORK_ROOT``, is the only
way the smoke exercises the real entry points with no path in common with a real
campaign.

Both cutters are deterministic (sorted order, round-robin over labels) and
idempotent: an output tree that already holds the requested number of items is
left alone, so a re-run of the smoke costs nothing here.

The CIPI cutter also rewrites the fold file
-------------------------------------------
``difficulty_splits_5fold.json`` names specific movement ids, and
``CIPIDifficultyEstimationEvalDataset.get_splits`` raises ``KeyError`` for any id
in a fold that is missing from ``difficulty_index.json``. So dropping movements
means dropping them from the folds too -- keeping that check meaningful instead
of disabling it. Movements are taken round-robin over the label values, so a
cut-down corpus still carries more than one class and the probe has something to
fit; the accuracy it produces is meaningless and is not meant to be read.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

#: Below this a MIDI file is almost certainly a stub the tokenizer will skip,
#: and a smoke that claims 100 files but trains on 60 is not measuring what it
#: says. Small enough to keep real-but-short pieces.
MIN_MIDI_BYTES = 1024


def cut_midi(src: Path, out: Path, n_files: int, n_shards: int) -> int:
    """Copy the first ``n_files`` MIDI files of ``src`` into ``n_shards`` shards.

    Shards are immediate subdirectories, which is exactly what
    ``run_tokenize.sh`` turns into array tasks -- so a smoke with two shards
    proves the array mapping as well as the tokenizer.
    """
    shards = [out / "shard{:02d}".format(i) for i in range(n_shards)]
    have = sum(len(list(s.glob("*"))) for s in shards if s.is_dir())
    if have >= n_files:
        print("[smoke-corpus] midi: {} files already under {}".format(have, out))
        return have

    if out.exists():
        shutil.rmtree(out)
    for s in shards:
        s.mkdir(parents=True)

    # os.walk sorted in place, not sorted(rglob("*")): the real corpus is ~1.7M
    # files, and materialising every path just to take the first hundred costs
    # minutes and hundreds of MB. Sorting each directory's own entries gives the
    # same deterministic order and stops as soon as enough files are found.
    picked = 0
    for parent, dirs, names in os.walk(src):
        dirs.sort()
        for name in sorted(names):
            if picked >= n_files:
                break
            path = Path(parent) / name
            if path.suffix.lower() not in (".mid", ".midi"):
                continue
            try:
                if not path.is_file() or path.stat().st_size < MIN_MIDI_BYTES:
                    continue
            except OSError:
                continue
            shutil.copy2(path, shards[picked % n_shards] / path.name)
            picked += 1
        if picked >= n_files:
            break

    if picked == 0:
        raise SystemExit("no MIDI file found under {} -- is $MIDI_DIR right?".format(src))
    print("[smoke-corpus] midi: {} files -> {} shards under {}".format(
        picked, n_shards, out))
    return picked


def cut_cipi(src: Path, out: Path, n_movements: int, target: str = "henle") -> int:
    """Copy ``n_movements`` CIPI movements, with folds and index cut to match."""
    index_path = src / "difficulty_cipi" / "metadata" / "difficulty_index.json"
    splits_path = src / "difficulty_cipi" / "splits" / "difficulty_splits_5fold.json"
    if not index_path.is_file() or not splits_path.is_file():
        raise SystemExit("no CIPI corpus at {} (want {} and {})".format(
            src, index_path, splits_path))

    dst = out / "difficulty_cipi"
    done = dst / "metadata" / "difficulty_index.json"
    if done.is_file() and len(json.loads(done.read_text())) >= n_movements:
        print("[smoke-corpus] cipi: {} movements already under {}".format(
            len(json.loads(done.read_text())), out))
        return len(json.loads(done.read_text()))

    index = json.loads(index_path.read_text())
    splits = json.loads(splits_path.read_text())
    midi_root = src / "difficulty_cipi" / "data" / "midi"

    # Only movements whose MIDI is actually on disk: the adapter builds its index
    # from the metadata and would hand the encoder a path that does not exist.
    have = {k: v for k, v in index.items()
            if (midi_root / (v["path"] + ".mid")).is_file()}

    by_label: dict[object, list[str]] = collections.defaultdict(list)
    for key, rec in have.items():
        by_label[rec[target]].append(key)
    for keys in by_label.values():
        keys.sort()

    keep: list[str] = []
    labels = sorted(by_label, key=str)
    rank = 0
    while len(keep) < n_movements:
        before = len(keep)
        for label in labels:
            if rank < len(by_label[label]) and len(keep) < n_movements:
                keep.append(by_label[label][rank])
        if len(keep) == before:
            break            # every label exhausted; the corpus is smaller than asked
        rank += 1
    kept = set(keep)

    # metadata order IS the dataset's positional index, so filter in place
    new_index = {k: v for k, v in index.items() if k in kept}
    new_splits = {fold: {name: {k: lab for k, lab in members.items() if k in kept}
                         for name, members in fold_splits.items()}
                  for fold, fold_splits in splits.items()}

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "metadata").mkdir(parents=True)
    (dst / "splits").mkdir(parents=True)
    (dst / "metadata" / "difficulty_index.json").write_text(
        json.dumps(new_index, indent=1))
    (dst / "splits" / "difficulty_splits_5fold.json").write_text(
        json.dumps(new_splits, indent=1))
    for rec in new_index.values():
        rel = rec["path"] + ".mid"
        target_path = dst / "data" / "midi" / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(midi_root / rel, target_path)

    sizes = {f: {n: len(m) for n, m in fs.items()} for f, fs in new_splits.items()}
    print("[smoke-corpus] cipi: {} movements, {} classes, folds {}".format(
        len(new_index), len({v[target] for v in new_index.values()}), sizes))
    return len(new_index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)

    m = sub.add_parser("midi", help="cut the pre-training corpus")
    m.add_argument("--src", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--files", type=int, default=100)
    m.add_argument("--shards", type=int, default=2)

    c = sub.add_parser("cipi", help="cut the CIPI evaluation corpus")
    c.add_argument("--src", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--movements", type=int, default=40)

    args = ap.parse_args()
    if args.what == "midi":
        cut_midi(Path(args.src), Path(args.out), args.files, max(1, args.shards))
    else:
        cut_cipi(Path(args.src), Path(args.out), args.movements)


if __name__ == "__main__":
    sys.exit(main())
