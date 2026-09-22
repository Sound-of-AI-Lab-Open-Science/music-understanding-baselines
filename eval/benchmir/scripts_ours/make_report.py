"""Turn benchmir's report.json into the consolidated markdown tables.

benchmir's own markdown writer raises NotImplementedError
(reporting/formats/markdown_writer.py:14), so the run emits json and this
script renders it. Rows are models, columns are tasks; folds are averaged and
the spread across folds is shown, because a single fold's number on CIPI or
POP909-chord (8 test songs per fold) is not a number anyone should quote.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# The package's one path configuration (published_code/pkgpaths.py). It reads
# paths.yaml and the same environment variables env.sh exports, so a default
# printed by --help is the path this script will actually use.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pkgpaths import PATHS  # noqa: E402

# The metric each task leads with, and the rest kept for the secondary table.
PRIMARY = {
    "cipi_difficulty": "accuracy",
    "emopia_emotion": "accuracy",
    "topmagd_genre": "accuracy",
    "pop909_chord": "chord_mirex",
    "pop909_root": "chord_root",
    "pop909_key": "key_weighted_score",
}
TASK_ORDER = list(PRIMARY)
# (task, rows, n_classes, majority-class fraction), measured on the extracted
# label vectors. Without these the accuracy column is unreadable: 0.29 is good
# on EMOPIA (4 classes, majority 0.289) and terrible on TopMAGD (13 classes,
# majority 0.601).
BASELINES = [
    ("cipi_difficulty", 569, 9, 0.216),
    ("emopia_emotion", 1071, 4, 0.289),
    ("topmagd_genre", 10282, 13, 0.601),
    ("pop909_chord", 18158, 77, 0.095),
    ("pop909_root", 18158, 13, 0.131),
    ("pop909_key", 74911, 26, 0.067),
]
# model_id -> the checkpoint it belongs to, so song/frame8/frame1 collapse into
# one row per model rather than three sparse ones.
def family(model_id: str) -> str:
    for suffix in ("-song", "-frame8", "-frame1", "-frame"):
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        blob = json.loads(p.read_text())
        rows.extend(blob["rows"] if isinstance(blob, dict) else blob)
    return rows


def agg(rows: list[dict]):
    """(model_family, task_id, metric) -> list of per-fold values."""
    out: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        for metric, value in r["metrics"].items():
            if value is None:
                continue
            out[(family(r["model_id"]), r["task_id"], metric)].append(float(value))
    return out


def cell(vals: list[float]) -> str:
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ±{statistics.pstdev(vals):.3f}"


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:-1]]
    out_path = Path(sys.argv[-1])
    rows = load(paths)
    data = agg(rows)

    models = sorted({m for m, _, _ in data})
    tasks = [t for t in TASK_ORDER if any(t == tt for _, tt, _ in data)]
    n_folds = {
        t: len({r["fold"] for r in rows if r["task_id"] == t}) for t in tasks
    }

    lines = ["# BenchMIR results — our three pretrained symbolic-music models", ""]
    lines.append("Cells are mean ±population SD across folds. Probe: MLP [256, 256], "
                 "dropout 0.2, Adam lr 1e-3, batch 64, 50 epochs, seed 42.")
    lines.append("")
    lines.append("Baselines for reading the accuracy column (whole corpus, "
                 "all folds pooled):")
    lines.append("")
    lines.append("| task | rows | classes | majority class | uniform chance |")
    lines.append("|---|---|---|---|---|")
    for t, rows_, k, maj in BASELINES:
        lines.append(f"| {t} | {rows_} | {k} | {maj:.3f} | {1/k:.3f} |")
    lines.append("")
    lines.append("## Primary metric per task")
    lines.append("")
    header = "| model | " + " | ".join(
        f"{t}<br>({PRIMARY[t]}, {n_folds[t]} fold{'s' if n_folds[t] != 1 else ''})"
        for t in tasks
    ) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(tasks) + 1))
    for m in models:
        cells = [cell(data.get((m, t, PRIMARY[t]), [])) for t in tasks]
        lines.append(f"| **{m}** | " + " | ".join(cells) + " |")

    lines += ["", "## All metrics", ""]
    all_metrics = sorted({(t, k) for _, t, k in data})
    lines.append("| task | metric | " + " | ".join(models) + " |")
    lines.append("|" + "---|" * (len(models) + 2))
    for t in tasks:
        for tt, metric in all_metrics:
            if tt != t:
                continue
            cells = [cell(data.get((m, t, metric), [])) for m in models]
            lines.append(f"| {t} | {metric} | " + " | ".join(cells) + " |")

    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
