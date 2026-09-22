"""Turn a corpus-quality CSV into the keep-list the union merge consumes.

A keep-list is one corpus id per line. ``merge_shard_caches.py --keep-ids``
intersects it with the already-built shard caches, so restricting the corpus
costs a minute of symlinking instead of a full re-tokenization -- which is the
whole reason the restriction is applied here and never in ``data.filter`` of a
training config (that feeds the cache fingerprint; see the note in
``configs/jepa_paper.yaml``).

The CSV is any table with a header row and one column naming the file, e.g.::

    file_path,n_notes,pitch_entropy,empty_measure_rate
    0/100120.mxl,812,3.41,0.02

The id is the basename with every extension stripped, which is exactly what
``make_musescore_split.py::id_of`` and the tokenization caches use -- so
``0/100120.mxl``, ``100120.mxl.mid`` and ``100120`` all collapse to ``100120``
and a CSV that names a different file extension than the corpus still matches.

Usage::

    python baselines/cache_tools/build_keep_list.py \\
        --csv quality_metrics.csv --column file_path \\
        --out "$WORK_ROOT/filter/ids_keep.txt"

    # keep only rows that pass a threshold
    python baselines/cache_tools/build_keep_list.py --csv q.csv \\
        --min n_notes=32 --max empty_measure_rate=0.5 --out ids_keep.txt

Every filter is optional: with none, the output is simply every id the CSV
names. The script reports how many rows it read, how many survived and how many
distinct ids that left, because "994,504 rows" and "994,460 distinct ids" are
not the same number and the difference is worth seeing before a run consumes it.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


def id_of(name: str) -> str:
    """``0/100120.mxl`` -> ``100120``; strips a directory and every extension."""
    stem = os.path.basename(name.strip().replace("\\", "/"))
    while True:
        root, ext = os.path.splitext(stem)
        if not ext:
            return stem
        stem = root


def _bounds(pairs: list[str], what: str) -> list[tuple[str, float]]:
    out = []
    for p in pairs or []:
        col, _, raw = p.partition("=")
        if not raw:
            raise SystemExit(f"--{what} needs column=value, got {p!r}")
        try:
            out.append((col, float(raw)))
        except ValueError:
            raise SystemExit(f"--{what} value must be a number, got {raw!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="quality/metadata CSV with a header row")
    ap.add_argument("--column", default="file_path",
                    help="column naming the file (default: file_path)")
    ap.add_argument("--out", required=True, help="keep-list to write, one id per line")
    ap.add_argument("--min", action="append", default=[], dest="mins",
                    help="column=value; drop rows below it (repeatable)")
    ap.add_argument("--max", action="append", default=[], dest="maxs",
                    help="column=value; drop rows above it (repeatable)")
    args = ap.parse_args()

    mins = _bounds(args.mins, "min")
    maxs = _bounds(args.maxs, "max")

    ids: list[str] = []
    seen: set[str] = set()
    n_rows = n_kept = n_unparsable = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if args.column not in (reader.fieldnames or []):
            raise SystemExit(
                f"column {args.column!r} is not in {args.csv}; "
                f"have {reader.fieldnames}")
        for col, _ in mins + maxs:
            if col not in (reader.fieldnames or []):
                raise SystemExit(f"threshold column {col!r} is not in {args.csv}")
        for row in reader:
            n_rows += 1
            try:
                if any(float(row[c]) < v for c, v in mins):
                    continue
                if any(float(row[c]) > v for c, v in maxs):
                    continue
            except (TypeError, ValueError):
                # A row whose metric is blank or non-numeric cannot be judged.
                # Dropping it silently would shrink the corpus invisibly, so it
                # is dropped loudly instead: counted and reported.
                n_unparsable += 1
                continue
            n_kept += 1
            i = id_of(row[args.column])
            if i and i not in seen:
                seen.add(i)
                ids.append(i)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + ("\n" if ids else ""))
    os.replace(tmp, args.out)   # write-then-rename: never a half-written keep-list

    print(f"rows read       : {n_rows}")
    print(f"rows kept       : {n_kept}")
    if n_unparsable:
        print(f"rows unjudgeable: {n_unparsable} (a threshold column was blank "
              f"or non-numeric)")
    print(f"distinct ids    : {len(ids)}")
    print(f"written         : {args.out}")
    if not ids:
        print("EMPTY keep-list -- the union merge would produce an empty cache",
              file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
