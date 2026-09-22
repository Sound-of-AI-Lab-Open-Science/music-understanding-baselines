"""Consolidate the ALL-909-song POP909 chord/root runs, alongside ours_v1's 30.

Same table shape as scripts_ours/make_report.py (primary metric per task, then
every metric, mean +-population SD over the 5 folds), plus a side-by-side table
against the 30-song ours_v1 numbers. Baselines are recomputed from the full
run's own extracted label vectors, because both the row count and the class
inventory change with the corpus (chord goes 77 -> 116 distinct labels).
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

FULL = PATHS.run_root / "pop909full"
V1 = PATHS.run_root / "v1"
PRIMARY = {"pop909_chord": "chord_mirex", "pop909_root": "chord_root"}
TASKS = list(PRIMARY)
MODELS = ["musetok", "music-jepa", "music-jepa-champA", "musicbert"]


def family(model_id: str) -> str:
    for suffix in ("-song", "-frame8", "-frame1", "-frame"):
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def load(root: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(root.glob("pop909_*/report.json")):
        blob = json.loads(p.read_text())
        rows.extend(blob["rows"] if isinstance(blob, dict) else blob)
    return [r for r in rows if r["task_id"] in TASKS]


def agg(rows: list[dict]) -> dict[tuple[str, str, str], list[float]]:
    out: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        for metric, value in r["metrics"].items():
            if value is not None:
                out[(family(r["model_id"]), r["task_id"], metric)].append(float(value))
    return out


def cell(vals: list[float]) -> str:
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ±{statistics.pstdev(vals):.3f}"


def baselines(root: Path) -> dict[str, tuple[str, int, str]]:
    """(rows, classes, majority fraction) per task, from the extracted blobs.

    Reported as a range over the four checkpoints, because the frame grid is the
    model's own bar segmentation: the label VOCABULARY is the corpus's, but how
    many frames carry each label is not identical across encoders (they disagree
    on a handful of pickup / meter-change bars). The spread is well under 1%.
    """
    out: dict[str, tuple[str, int, str]] = {}
    for task in TASKS:
        ns, majs, ks = [], [], set()
        for m in MODELS:
            blobs = sorted(root.glob(f"{task}__{m}*/cache/embeddings/*.pt"))
            if not blobs:
                continue
            labels = torch.load(
                blobs[0], map_location="cpu", weights_only=False
            )["labels"]
            counts = Counter(int(x) for x in labels.tolist())
            ns.append(len(labels))
            majs.append(max(counts.values()) / len(labels))
            ks.add(len(counts))
        if not ns:
            continue
        rows = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}–{max(ns)}"
        maj = (
            f"{majs[0]:.3f}"
            if abs(max(majs) - min(majs)) < 5e-4
            else f"{min(majs):.3f}–{max(majs):.3f}"
        )
        out[task] = (rows, max(ks), maj)
    return out


def main() -> None:
    full_rows, v1_rows = load(FULL), load(V1)
    full, v1 = agg(full_rows), agg(v1_rows)
    base_full, base_v1 = baselines(FULL), baselines(V1)
    models = [m for m in MODELS if any(m == mm for mm, _, _ in full)]
    n_folds = {
        t: len({r["fold"] for r in full_rows if r["task_id"] == t}) for t in TASKS
    }

    L = ["# POP909 chord / root on ALL 909 songs", ""]
    L += [
        "Same probe protocol as `ours_v1`: MLP [256, 256], dropout 0.2, Adam lr 1e-3,",
        "batch 64, 50 epochs, seed 42, 5 folds, `frames_per_bar: 8`, identical",
        "checkpoints. The only change is `max_songs: all` on the dataset, which lifts",
        "the historical `[:30]` slice in `pop909cl_eval_corpus._build_index()`.",
        "ours_v1's configs are untouched and its 30-song numbers still reproduce.",
        "",
        "Cells are mean ±population SD across the 5 folds.",
        "",
        "Runtime, measured on the 40 per-fold SLURM CPU jobs (8 threads,",
        "64 GB): frame extraction from the content-hash bar cache",
        "72 s per job for ~600k rows; probe fit+eval 1085-2584 s per fold",
        "(mean 1920 s, 21.3 CPU-hours over the 40 jobs). Peak RSS ~4.1 GB, so",
        "the 64 GB request is generous; the 1.85 GB embedding tensor dominates.",
        "",
        "## Baselines (whole corpus, all folds pooled)",
        "",
        "| task | songs | rows | classes | majority class | uniform chance |",
        "|---|---|---|---|---|---|",
    ]
    for t in TASKS:
        for label, b in (("30", base_v1.get(t)), ("909", base_full.get(t))):
            if b:
                n, k, maj = b
                L.append(f"| {t} | {label} | {n} | {k} | {maj} | {1/k:.3f} |")
    L += ["", "## Primary metric per task (909 songs)", ""]
    L.append(
        "| model | "
        + " | ".join(f"{t}<br>({PRIMARY[t]}, {n_folds[t]} folds)" for t in TASKS)
        + " |"
    )
    L.append("|" + "---|" * (len(TASKS) + 1))
    for m in models:
        L.append(
            f"| **{m}** | "
            + " | ".join(cell(full.get((m, t, PRIMARY[t]), [])) for t in TASKS)
            + " |"
        )

    L += ["", "## All metrics (909 songs)", ""]
    L.append("| task | metric | " + " | ".join(models) + " |")
    L.append("|" + "---|" * (len(models) + 2))
    for t in TASKS:
        for metric in sorted({k for _, tt, k in full if tt == t}):
            L.append(
                f"| {t} | {metric} | "
                + " | ".join(cell(full.get((m, t, metric), [])) for m in models)
                + " |"
            )

    L += ["", "## 30 songs (ours_v1) vs 909 songs, side by side", ""]
    L.append("| task | metric | model | 30 songs | 909 songs | delta |")
    L.append("|---|---|---|---|---|---|")
    for t in TASKS:
        for metric in sorted({k for _, tt, k in full if tt == t}):
            for m in models:
                a, b = v1.get((m, t, metric), []), full.get((m, t, metric), [])
                d = (
                    f"{statistics.mean(b) - statistics.mean(a):+.3f}"
                    if a and b
                    else "--"
                )
                L.append(f"| {t} | {metric} | {m} | {cell(a)} | {cell(b)} | {d} |")

    out = FULL / "CONSOLIDATED.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
