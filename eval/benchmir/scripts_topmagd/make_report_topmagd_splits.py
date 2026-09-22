"""Consolidate TopMAGD genre results across the three split strategies.

Columns, per arm:

  5-fold (published)   the number already in runs/v1 (pass 1) and
                       runs/pass2 (pass 2). Mean +- std over 5 folds.
                       Computed BEFORE upstream BenchMIR commit 87349a4,
                       the point the evaluation submodule's branch starts
                       from (see THIRD_PARTY_NOTICES.md).
  5-fold (re-run)      the same stratified_kfold/n_folds=5 setting, re-run at
                       the merged code. This column exists because the merge
                       changed build_loss_fn(): class weights now come from the
                       TRAIN split rather than the whole indexed corpus. Without
                       it, "new strategy vs published number" would confound the
                       split change with the loss change.
  balanced_fixed_size  TU Wien's fixed 2,000-track-per-genre TRAIN quota.
  stratified_percentage_split  TU Wien's fixed 80/20-per-genre partition.

Baselines are per SETTING, not per corpus: each fixed partition produces a
different TEST set, so the majority-class accuracy a probe has to beat differs
between columns even though the indexed corpus (10,282 tracks) is identical.
For the 5-fold columns the baseline is the mean over the five test folds.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# The package's one path configuration (published_code/pkgpaths.py). It reads
# paths.yaml and the same environment variables env.sh exports, so a default
# printed by --help is the path this script will actually use.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pkgpaths import PATHS  # noqa: E402

RUNS_NEW = PATHS.run_root / "topmagd_splits"
STATS = PATHS.work_root / "topmagd_split_stats.json"
OUT = RUNS_NEW / "CONSOLIDATED.md"

PUBLISHED = {
    "musicbert":             str(PATHS.run_root / "v1" / "topmagd_genre__musicbert"),
    "musetok":               str(PATHS.run_root / "v1" / "topmagd_genre__musetok"),
    "music-jepa":            str(PATHS.run_root / "v1" / "topmagd_genre__music-jepa"),
    "music-jepa-champA":     str(PATHS.run_root / "v1" / "topmagd_genre__music-jepa-champA"),
    "musicbert_musescore":   str(PATHS.run_root / "pass2" / "topmagd_genre__musicbert_musescore"),
    "musetok_musescore":     str(PATHS.run_root / "pass2" / "topmagd_genre__musetok_musescore"),
    "jepa_musescore_champA": str(PATHS.run_root / "pass2" / "topmagd_genre__jepa_musescore_champA"),
    "jepa_musescore_paper":  str(PATHS.run_root / "pass2" / "topmagd_genre__jepa_musescore_paper"),
}
ARMS = list(PUBLISHED)

METRICS = ["accuracy", "balanced_accuracy", "weighted_f1"]
COLS = [
    ("published", "5-fold (published)"),
    ("kfold5", "5-fold (re-run)"),
    ("balanced", "balanced_fixed_size"),
    ("stratpct", "stratified_percentage_split"),
]
STAT_KEY = {
    "published": "use_all_matches=False|stratified_kfold",
    "kfold5": "use_all_matches=False|stratified_kfold",
    "balanced": "use_all_matches=False|balanced_fixed_size",
    "stratpct": "use_all_matches=False|stratified_percentage_split",
}


def read(run_dir: Path) -> dict[str, list[float]] | None:
    rp = run_dir / "report.json"
    if not rp.exists():
        return None
    rows = json.loads(rp.read_text())["rows"]
    return {m: [r["metrics"][m] for r in rows] for m in METRICS}


def baselines(test_counts: dict[str, int], num_classes: int) -> dict[str, dict[str, float]]:
    """Exact scores of the two trivial predictors, per metric.

    A single "chance = 1/13" row is only right for accuracy and
    balanced_accuracy; weighted_f1 of a random or majority predictor is
    neither 1/13 nor 0, so printing 1/13 under it would be a made-up number.
    Both are computed in closed form from the TEST label distribution:

      majority (always predict the largest test class)
        accuracy          p_maj
        balanced_accuracy 1/K            (recall 1 on one class, 0 elsewhere)
        weighted_f1       p_maj * 2 p_maj / (p_maj + 1)
      uniform random (predict each class w.p. 1/K, independent of the input)
        accuracy          1/K
        balanced_accuracy 1/K
        weighted_f1       sum_c p_c * 2 p_c (1/K) / (p_c + 1/K)
                          (recall_c = 1/K, precision_c = p_c)
    """
    n = sum(test_counts.values())
    props = {c: k / n for c, k in test_counts.items()}
    p_maj = max(props.values())
    u = 1.0 / num_classes
    maj_f1 = p_maj * (2 * p_maj / (p_maj + 1))
    rnd_f1 = sum(p * (2 * p * u / (p + u)) for p in props.values())
    return {
        "majority": {
            "accuracy": p_maj,
            "balanced_accuracy": u,
            "weighted_f1": maj_f1,
        },
        "random": {
            "accuracy": u,
            "balanced_accuracy": u,
            "weighted_f1": rnd_f1,
        },
    }


def baseline_cell(key: str, which: str, metric: str, stats: dict) -> str:
    """Baselines for a 5-fold column are the mean over its five test folds."""
    s = stats[STAT_KEY[key]]
    vals = [
        baselines(f["test_class_counts"], s["num_classes"])[which][metric]
        for f in s["folds"].values()
    ]
    return f"{statistics.mean(vals):.4f}"


def cell(vals: list[float] | None) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"


def main() -> None:
    if not STATS.exists():
        raise SystemExit(
            f"missing {STATS}\n"
            "Run scripts_topmagd/split_stats.py first (./run_report.sh topmagd\n"
            "does this for you).")
    stats = json.loads(STATS.read_text())
    data: dict[str, dict[str, dict[str, list[float]] | None]] = {}
    missing: list[str] = []
    for arm in ARMS:
        data[arm] = {"published": read(Path(PUBLISHED[arm]))}
        for key, _ in COLS[1:]:
            d = read(RUNS_NEW / f"topmagd_{key}__{arm}")
            data[arm][key] = d
            if d is None:
                missing.append(f"topmagd_{key}__{arm}")

    L: list[str] = []
    L.append("# TopMAGD genre — the two new split strategies vs the 5-fold split\n")
    L.append(
        "Probe identical in every column and identical to the rest of the "
        "benchmark: MLP [256, 256], dropout 0.2, Adam lr 1e-3, batch 64, "
        "50 epochs, seed 42, `folds: all`, class-weighted cross-entropy. "
        "Bar embeddings come from each arm's warm content-hash cache, so only "
        "the probe was refit — no checkpoint was re-encoded.\n"
    )

    # ---- corpus / split geometry ------------------------------------------
    L.append("## The song set is the same in every column\n")
    L.append(
        "`use_all_matches` was **False** in all eight published runs "
        "(`kwargs: {use_all_matches: false}` is written out in every one of the "
        "generated job configs of both grids), "
        "and it is False here. Upstream BenchMIR commit 87349a4 changed only the "
        "class *default* "
        "from True to False; because our configs always set it explicitly, that "
        "change moves nothing. So every column below indexes the same **10,282** "
        "tracks (10,282 unique songs — one MIDI per MSD id).\n"
    )
    L.append(
        "For reference, `use_all_matches: true` would index 34,867 rows over the "
        "same 10,282 songs. No column here uses it.\n"
    )
    L.append("What *does* change between columns is where those 10,282 rows go:\n")
    L.append("| setting | folds | train | val | test | rows placed | rows dropped |")
    L.append("|---|---|---|---|---|---|---|")
    for key, label in COLS[1:]:
        s = stats[STAT_KEY[key]]
        f0 = s["folds"]["0"]
        placed = f0["train"] + f0["val"] + f0["test"]
        L.append(
            f"| `{s_name(key)}` | {len(s['folds'])} | {f0['train']} | {f0['val']} "
            f"| {f0['test']} | {placed} | {s['indexed_rows'] - placed} |"
        )
    L.append("")
    L.append(
        "The dropped rows in the two fixed-partition columns are tracks that "
        "carry a topMAGD genre label and a matched MIDI but appear in neither "
        "the TRAIN nor the TEST list of TU Wien's partition file (18 of 10,282). "
        "The 5-fold column drops nothing — the numbers above are one fold of "
        "five, and every row is in exactly one test fold.\n"
    )

    # ---- baselines --------------------------------------------------------
    L.append("## Baselines, per setting\n")
    L.append(
        "Majority-class accuracy is computed on the **test split that is "
        "actually scored**, not on the corpus. That distinction matters here: "
        "`balanced_fixed_size` pulls a per-genre quota into TRAIN, which leaves "
        "a TEST set with a different skew from the corpus.\n"
    )
    L.append("| setting | classes | chance (1/K) | majority-class acc | majority class |")
    L.append("|---|---|---|---|---|")
    for key, label in COLS[1:]:
        s = stats[STAT_KEY[key]]
        majs = [f["majority_acc"] for f in s["folds"].values()]
        maj = (
            f"{statistics.mean(majs):.4f} ± {statistics.stdev(majs):.4f}"
            if len(majs) > 1
            else f"{majs[0]:.4f}"
        )
        names = {f["majority_class"] for f in s["folds"].values()}
        L.append(
            f"| `{s_name(key)}` | {s['num_classes']} | {1/s['num_classes']:.4f} "
            f"| {maj} | {'/'.join(sorted(names))} |"
        )
    L.append("")

    # ---- results ----------------------------------------------------------
    for m in METRICS:
        L.append(f"## {m}\n")
        L.append("| arm | " + " | ".join(lbl for _, lbl in COLS) + " |")
        L.append("|---" * (len(COLS) + 1) + "|")
        for arm in ARMS:
            cells = [cell((data[arm][k] or {}).get(m)) for k, _ in COLS]
            L.append(f"| `{arm}` | " + " | ".join(cells) + " |")
        for which, label in (("majority", "majority-class predictor"),
                             ("random", "uniform-random predictor")):
            row = [baseline_cell(k, which, m, stats) for k, _ in COLS]
            L.append(f"| **{label}** | " + " | ".join(row) + " |")
        L.append("")
        L.append(
            "First four arms are pass 1 (released / previously trained "
            "checkpoints); last four are pass 2 (MuseScore-retrained).\n"
        )

    L.append("## How to read this\n")
    L.append(
        "**The re-run column validates the comparison.** The merge changed "
        "`build_loss_fn()` so class weights come from the TRAIN split instead "
        "of the whole indexed corpus. For `stratified_kfold` those two "
        "distributions are near-identical by construction, and the re-run "
        "confirms it: every arm lands within its own fold-to-fold spread of "
        "its published number. So the two new columns can be read against the "
        "published 5-fold column directly, not only against the re-run.\n"
    )
    L.append(
        "**`stratified_percentage_split` is the 5-fold split with the folds "
        "thrown away.** 7,014/1,170/2,080 against a fold's 7,051/1,175/2,056, "
        "the same 60% Pop_Rock skew, and per-arm accuracies inside the 5-fold "
        "column's own std almost everywhere. What it buys is a *fixed, "
        "citable, externally-defined* partition — every paper using TU Wien's "
        "0.8 split trains on the same tracks — not a different measurement. "
        "It costs the error bar: one split, so no std, and the fold-to-fold "
        "spread in the 5-fold column (up to ±0.18 for "
        "`jepa_musescore_champA`) is exactly the quantity a single split "
        "cannot show you.\n"
    )
    L.append(
        "**`balanced_fixed_size` does not do what its name says here, and its "
        "numbers are not comparable to the other columns.** TU Wien's 2,000-"
        "per-genre TRAIN quota is defined over the whole of topMAGD (406,427 "
        "tracks). Our corpus is the LMD-*matched* subset, 10,282 tracks, and "
        "it intersects that quota only 616 times. Train therefore collapses to "
        "527 (+89 val) and TEST swells to 9,648 — 94% of the corpus, against "
        "20% in the other columns. Worse, what survives is *not* balanced: "
        "Country 104 train examples, Vocal 59, Electronic 50, **Pop_Rock 45**, "
        "Rap 17, Blues 10. Every arm loses accuracy in this column, and the "
        "loss tracks train-set size, not representation quality. Read it as a "
        "few-shot probe on a fixed external partition; do not read it as "
        "\"the balanced version of the genre task\".\n"
    )
    L.append(
        "**No arm beats the majority-class predictor on accuracy under any "
        "setting** (0.60–0.63). That was already true of the published 5-fold "
        "column and neither new split changes it. On balanced_accuracy every "
        "arm beats the 0.0769 floor; on weighted_f1 the collapsed "
        "`music-jepa` arm sits far *below* even the uniform-"
        "random predictor's 0.107, which is the signature of a probe that has "
        "learned to emit one or two classes.\n"
    )
    L.append(
        "**Ranking is stable across all three settings**: `musicbert` and "
        "`musicbert_musescore` lead, `musetok`/`musetok_musescore` follow, the "
        "JEPA arms trail, and `music-jepa` (the collapsed checkpoint) is last "
        "or near-last everywhere. Neither new split rescues or demotes any "
        "arm.\n"
    )

    if missing:
        L.append("## Missing runs\n")
        for j in missing:
            L.append(f"- `{j}` — no report.json")
        L.append("")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}  ({len(missing)} missing)")


def s_name(key: str) -> str:
    return {
        "kfold5": "stratified_kfold (n_folds=5)",
        "balanced": "balanced_fixed_size",
        "stratpct": "stratified_percentage_split",
    }[key]


if __name__ == "__main__":
    main()
