"""CIPI difficulty on the official movement-level tree, beside pass 1's numbers.

The two columns are NOT comparable and the table says so at the top rather than
in a footnote. `runs/v1` scored a piece-level index (569 rows, ids `c-1`);
this scores the official movement-level `difficulty_cipi` (592 rows, ids
`c-11::3`). The id sets do not intersect, so the 5-fold splits are entirely
different -- a model can move in this table because the folds moved, and
nothing here can separate that from a real change. What the pair IS good for is
the ordering of the four checkpoints, which is a within-column question.

Baselines (rows / classes / majority) are recomputed from each run's own
extracted label vectors, not carried over.

Usage:  make_report_cipi_official.py [out.md]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

# The package's one path configuration (published_code/pkgpaths.py). It reads
# paths.yaml and the same environment variables env.sh exports, so a default
# printed by --help is the path this script will actually use.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pkgpaths import PATHS  # noqa: E402

NEW = PATHS.run_root / "cipi_official"
OLD = PATHS.run_root / "v1"
TASK = "cipi_difficulty"
ARMS = ["musicbert", "musetok", "music-jepa", "music-jepa-champA"]
METRICS = ["accuracy", "balanced_accuracy", "balanced_within_one_accuracy",
           "macro_f1", "ordinal_mae"]
#: ordinal_mae is an error: lower is better, so it must not be ranked with the rest.
LOWER_IS_BETTER = {"ordinal_mae"}


def family(model_id: str) -> str:
    for suffix in ("-song", "-frame8", "-frame1", "-frame"):
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def load(root: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(root.glob(f"{TASK}*/report.json")):
        blob = json.loads(p.read_text())
        rows.extend(blob["rows"] if isinstance(blob, dict) else blob)
    return [r for r in rows if r["task_id"] == TASK]


def agg(rows: list[dict]) -> dict:
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        for metric, value in r["metrics"].items():
            if value is not None:
                out[(family(r["model_id"]), metric)].append(float(value))
    return out


def cell(vals: list[float]) -> str:
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ±{statistics.pstdev(vals):.3f}"


def baseline(root: Path) -> tuple[str, int, str]:
    """(rows, classes, majority fraction) from the extracted label vectors."""
    ns, majs, ks = [], [], set()
    for arm in ARMS:
        blobs = sorted(root.glob(f"{TASK}__{arm}*/cache/embeddings/*.pt"))
        if not blobs:
            continue
        labels = torch.load(blobs[0], map_location="cpu", weights_only=False)["labels"]
        counts = Counter(int(x) for x in labels.tolist())
        ns.append(len(labels))
        majs.append(max(counts.values()) / len(labels))
        ks.add(len(counts))
    if not ns:
        return ("--", 0, "--")
    rows = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}-{max(ns)}"
    maj = (f"{majs[0]:.3f}" if abs(max(majs) - min(majs)) < 5e-4
           else f"{min(majs):.3f}-{max(majs):.3f}")
    return (rows, max(ks), maj)


def ranking(data: dict, metric: str) -> list[str]:
    have = [(a, statistics.mean(data[(a, metric)]))
            for a in ARMS if data.get((a, metric))]
    return [a for a, _ in sorted(have, key=lambda kv: kv[1],
                                 reverse=metric not in LOWER_IS_BETTER)]


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else NEW / "CONSOLIDATED.md"
    new_rows, old_rows = load(NEW), load(OLD)
    if not new_rows:
        raise SystemExit(f"no {TASK} reports under {NEW}")
    new, old = agg(new_rows), agg(old_rows)
    bn, bo = baseline(NEW), baseline(OLD)
    arms = [a for a in ARMS if any(a == m for m, _ in new)]

    L = ["# CIPI difficulty — official movement-level `difficulty_cipi`", ""]
    L += [
        "**The two columns are not comparable.** `runs/v1` scored a",
        "hand-built PIECE-level index (ids like `c-1`); this run scores the",
        "official `difficulty_cipi` distribution, which is",
        "MOVEMENT-level (ids like `c-11::3`). The two id sets do not intersect at",
        "all, so the 5-fold splits are entirely different (350/118/124 per fold",
        "here). A number can move between the columns because the folds moved, and",
        "nothing in this table separates that from a change in the model. Henle",
        "labels agree on the pieces both trees cover, so this is a change of unit,",
        "not of ground truth. Read each column against its own baselines, and",
        "compare the two only on the ORDERING of the four checkpoints.",
        "",
        "Checkpoints are pass 1's four, unchanged — the released / previously",
        "trained weights, not the MuseScore retrains. Probe: MLP [256, 256],",
        "dropout 0.2, Adam lr 1e-3, batch 64, 50 epochs, seed 42, 5 folds.",
        "Cells are mean ±population SD across the 5 folds.",
        "",
        "## Baselines (whole corpus, all folds pooled)",
        "",
        "| index | rows | classes | majority class | uniform chance |",
        "|---|---|---|---|---|",
    ]
    for label, (r, k, maj) in (("official (movement-level)", bn),
                               ("ours_v1 (piece-level)", bo)):
        chance = f"{1/k:.3f}" if k else "--"
        L.append(f"| {label} | {r} | {k or '--'} | {maj} | {chance} |")

    L += ["", "## Side by side", ""]
    L.append("| metric | " + " | ".join(
        f"{a}<br>official | {a}<br>ours_v1" for a in arms) + " |")
    L.append("|" + "---|" * (2 * len(arms) + 1))
    for metric in METRICS:
        cells = []
        for a in arms:
            cells.append(cell(new.get((a, metric), [])))
            cells.append(cell(old.get((a, metric), [])))
        arrow = " (lower is better)" if metric in LOWER_IS_BETTER else ""
        L.append(f"| {metric}{arrow} | " + " | ".join(cells) + " |")

    L += ["", "## Does the ranking change?", "",
          "| metric | official (best first) | ours_v1 (best first) | same? |",
          "|---|---|---|---|"]
    for metric in METRICS:
        rn, ro = ranking(new, metric), ranking(old, metric)
        same = "yes" if rn == ro and rn else ("--" if not ro else "**no**")
        L.append(f"| {metric} | {' > '.join(rn) or '--'} | "
                 f"{' > '.join(ro) or '--'} | {same} |")
    L.append("")

    out_path.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
