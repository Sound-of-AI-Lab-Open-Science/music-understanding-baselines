"""Consolidate the pass-2 grid into CONSOLIDATED.md.

Same table shape as scripts_ours/make_report.py -- primary metric per task,
then every metric, mean ±population SD over folds -- extended to pass 2's eight
task ids (POP909 chord/root appear twice, on the 30-song slice and on all 909)
and four arms.

Baselines are computed from each run's OWN extracted label vectors rather than
hardcoded, because the row count and the class inventory both depend on the
corpus slice: POP909 chord is 77 labels over ~18k rows at 30 songs and 116 over
~550k at 909.

Usage:
    make_report_pass2.py <runs_dir> <out.md> [--title "..."] [--note "..."]
"""

from __future__ import annotations

import argparse
import json
import os
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

PRIMARY = {
    "cipi_difficulty": "accuracy",
    "emopia_emotion": "accuracy",
    "topmagd_genre": "accuracy",
    "pop909_chord": "chord_mirex",
    "pop909_root": "chord_root",
    "pop909_chord_full": "chord_mirex",
    "pop909_root_full": "chord_root",
    "pop909_key": "key_weighted_score",
}
TASK_ORDER = list(PRIMARY)
ARMS = ["musicbert", "jepa_musescore_paper", "jepa_musescore_champA",
        "musetok_musescore", "musicbert_musescore"]


def family(model_id: str) -> str:
    for suffix in ("-song", "-frame8", "-frame1", "-frame"):
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def load(root: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(root.glob("*/report.json")):
        blob = json.loads(p.read_text())
        rows.extend(blob["rows"] if isinstance(blob, dict) else blob)
    return rows


def agg(rows: list[dict]) -> dict:
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


def baselines(root: Path, tasks: list[str]) -> dict:
    """(rows, classes, majority fraction) per task, from the extracted blobs.

    Reported as a range over the arms, because the frame grid is the model's own
    bar segmentation: the label VOCABULARY is the corpus's, but how many frames
    carry each label is not identical across encoders (they disagree on a
    handful of pickup / meter-change bars).
    """
    out: dict[str, tuple[str, int, str]] = {}
    for task in tasks:
        ns, majs, ks = [], [], set()
        for arm in ARMS:
            blobs = sorted(root.glob(f"{task}__{arm}*/cache/embeddings/*.pt"))
            if not blobs:
                continue
            labels = torch.load(blobs[0], map_location="cpu",
                                weights_only=False)["labels"]
            counts = Counter(int(x) for x in labels.tolist())
            ns.append(len(labels))
            majs.append(max(counts.values()) / len(labels))
            ks.add(len(counts))
        if not ns:
            continue
        rows = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}-{max(ns)}"
        maj = (f"{majs[0]:.3f}" if abs(max(majs) - min(majs)) < 5e-4
               else f"{min(majs):.3f}-{max(majs):.3f}")
        out[task] = (rows, max(ks), maj)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("out")
    ap.add_argument("--title", default="BenchMIR pass 2")
    ap.add_argument("--note", action="append", default=[],
                    help="a paragraph inserted under the title (repeatable)")
    args = ap.parse_args()

    root = Path(args.runs_dir)
    rows = load(root)
    if not rows:
        raise SystemExit(f"no */report.json under {root}")
    data = agg(rows)
    models = [m for m in ARMS if any(m == mm for mm, _, _ in data)]
    models += sorted({m for m, _, _ in data} - set(models))
    tasks = [t for t in TASK_ORDER if any(t == tt for _, tt, _ in data)]
    n_folds = {t: len({r["fold"] for r in rows if r["task_id"] == t}) for t in tasks}
    base = baselines(root, tasks)

    L = [f"# {args.title}", ""]
    for note in args.note:
        L += [note, ""]
    L += ["Cells are mean ±population SD across folds. Probe: MLP [256, 256], "
          "dropout 0.2, Adam lr 1e-3, batch 64, 50 epochs, seed 42.", ""]

    # Every launch drops a _checkpoints.txt in its own jobs directory --
    # per-arm launches each get one -- so the table is merged from all of them.
    # A pass that cannot say which weights produced it is not a pass.
    jobs_root = Path(os.environ.get(
        "BENCHMIR_PASS2_JOBS_ROOT", str(root.parent.parent / "jobs_pass2")))
    # Ordering matters and is not alphabetical. A per-arm launch writes
    # jobs_pass2/<arm>/_checkpoints.txt, and that file is authoritative FOR THAT
    # ARM; every other _checkpoints.txt is from some other launch and may name a
    # stale checkpoint for it. Reading them in sorted order let a directory
    # called `snapshot_tail` overwrite the champ_A row with the snapshot's
    # checkpoint -- a table that named the wrong weights for the numbers beside
    # them. So: generic files first, then each arm's own directory last, and
    # only arms that actually have reports are listed.
    seen: dict[str, str] = {}
    generic = [c for c in sorted(jobs_root.glob("**/_checkpoints.txt"))
               if c.parent.name not in ARMS]
    per_arm = [jobs_root / a / "_checkpoints.txt" for a in ARMS]
    for ck in generic + [c for c in per_arm if c.is_file()]:
        for line in ck.read_text().splitlines():
            if line.strip():
                arm, _, path = line.partition("\t")
                if ck.parent.name in ARMS and ck.parent.name != arm:
                    continue      # an arm dir speaks only for its own arm
                seen[arm] = path
    listed = [a for a in models if a in seen]
    if listed:
        L += ["Checkpoints evaluated:", "", "| arm | checkpoint |", "|---|---|"]
        for arm in listed:
            L.append(f"| `{arm}` | `{seen[arm]}` |")
        L.append("")

    if base:
        L += ["Baselines for reading the accuracy column (whole corpus, all "
              "folds pooled), measured on each run's own extracted labels:", "",
              "| task | rows | classes | majority class | uniform chance |",
              "|---|---|---|---|---|"]
        for t in tasks:
            if t not in base:
                continue
            r, k, maj = base[t]
            L.append(f"| {t} | {r} | {k} | {maj} | {1/k:.3f} |")
        L.append("")

    missing = [a for a in ARMS if a not in models]
    if missing:
        L += [f"Arms not yet in this table: {', '.join('`'+a+'`' for a in missing)}. "
              "Each arm is launched and consolidated independently, so re-running "
              "this script after another arm finishes adds its column without "
              "touching the ones already here.", ""]

    L += ["## Primary metric per task", ""]
    L.append("| model | " + " | ".join(
        f"{t}<br>({PRIMARY[t]}, {n_folds[t]} fold{'s' if n_folds[t] != 1 else ''})"
        for t in tasks) + " |")
    L.append("|" + "---|" * (len(tasks) + 1))
    for m in models:
        L.append(f"| **{m}** | " + " | ".join(
            cell(data.get((m, t, PRIMARY[t]), [])) for t in tasks) + " |")

    L += ["", "## All metrics", ""]
    all_metrics = sorted({(t, k) for _, t, k in data})
    L.append("| task | metric | " + " | ".join(models) + " |")
    L.append("|" + "---|" * (len(models) + 2))
    for t in tasks:
        for tt, metric in all_metrics:
            if tt == t:
                L.append(f"| {t} | {metric} | " + " | ".join(
                    cell(data.get((m, t, metric), [])) for m in models) + " |")

    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
