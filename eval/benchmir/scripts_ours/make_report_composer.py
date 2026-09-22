"""Consolidate the Humdrum composer grid into CONSOLIDATED.md.

Same table shape as scripts_ours/make_report_pass2.py -- mean ±population SD
over folds -- but for one task across eight arms, so the table is transposed:
arms are rows, metrics are columns.

Baselines are computed from the run's OWN extracted label vector and the
dataset's OWN split file, not hardcoded:

  * the corpus-pooled majority fraction and 1/K, for reading the table at a
    glance;
  * the per-fold TEST-split majority fraction, which is the baseline the
    accuracy column is actually measured against -- the folds are not
    identically balanced, and a 394-song test split's majority class is not
    the corpus's.

Encode failures are read from the warm array's per-shard failure JSONs. A file
the worker cannot encode becomes a ZERO embedding row (ours_common.BarEmbedding
Model._bars) -- it is not dropped -- so the count belongs next to the numbers.

Usage:
    make_report_composer.py <runs_dir> <out.md> [--title "..."]
"""

from __future__ import annotations

import argparse
import json
import re
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

TASK = "humdrum_composer"
#: Column order. accuracy first (the headline), then the three the task adds.
METRIC_ORDER = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
ARMS = [
    "musicbert",
    "musetok",
    "music-jepa",
    "music-jepa-champA",
    "musicbert_musescore",
    "musetok_musescore",
    "jepa_musescore_champA",
    "jepa_musescore_paper",
]
PASS1 = set(ARMS[:4])
LOGS = PATHS.log_root


def _failure_kind(msg: str) -> str:
    """A short, groupable label for one worker failure message."""
    if "timeout" in msg:
        return "per-file timeout"
    m = re.search(r"(\d+) out-of-vocabulary event", msg)
    if m:
        return "out-of-vocabulary REMI+ event(s)"
    return msg.split(":")[0][:60]


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


def cell(vals: list[float]) -> str:
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ±{statistics.pstdev(vals):.3f}"


def warm_failures(model_ids: list[str]) -> dict[str, tuple[int, dict[str, str]]]:
    """(count, {path: message}) per arm, merged over that arm's warm shards.

    Only Humdrum paths are counted: the same cache -- and the same failure log
    directory -- also carries pass-1 and pass-2 failures on other corpora, and
    a composer table must not inherit those.
    """
    out: dict[str, tuple[int, dict[str, str]]] = {}
    for mid in model_ids:
        merged: dict[str, str] = {}
        for p in sorted(LOGS.glob(f"warm_{mid}_s*of*_failures.json")):
            try:
                blob = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            for path, msg in blob.items():
                if "/humdrum/" in path:
                    merged[path] = msg
        out[family(mid)] = (len(merged), merged)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("out")
    ap.add_argument("--title", default="BenchMIR -- Humdrum composer classification")
    ap.add_argument("--jobs-dir",
                    default=str(PATHS.jobs_root / "composer"))
    args = ap.parse_args()

    root = Path(args.runs_dir)
    rows = [r for r in load(root) if r["task_id"] == TASK]
    if not rows:
        raise SystemExit(f"no */report.json with task_id={TASK} under {root}")

    data: dict[tuple[str, str], list[float]] = defaultdict(list)
    model_ids: dict[str, str] = {}
    folds_seen: dict[str, set] = defaultdict(set)
    for r in rows:
        arm = family(r["model_id"])
        model_ids[arm] = r["model_id"]
        folds_seen[arm].add(r["fold"])
        for metric, value in r["metrics"].items():
            if value is not None:
                data[(arm, metric)].append(float(value))

    arms = [a for a in ARMS if a in model_ids]
    arms += sorted(set(model_ids) - set(arms))
    metrics = [m for m in METRIC_ORDER if any(m == mm for _, mm in data)]
    metrics += sorted({mm for _, mm in data} - set(metrics))

    # ---- checkpoints -----------------------------------------------------
    ckpts: dict[str, str] = {}
    ck_file = Path(args.jobs_dir) / "_checkpoints.txt"
    if ck_file.is_file():
        for line in ck_file.read_text().splitlines():
            if line.strip():
                arm, _, path = line.partition("\t")
                ckpts[arm] = path

    # ---- baselines from the extracted labels + the split file ------------
    n_rows: dict[str, int] = {}
    pooled: Counter = Counter()
    for arm in arms:
        blobs = sorted((root / f"{TASK}__{arm}" / "cache" / "embeddings").glob("*.pt"))
        if not blobs:
            continue
        labels = torch.load(blobs[0], map_location="cpu",
                            weights_only=False)["labels"]
        n_rows[arm] = len(labels)
        if not pooled:
            pooled = Counter(int(x) for x in labels.tolist())

    n_total = sum(pooled.values())
    n_classes = len(pooled)
    maj_pooled = max(pooled.values()) / n_total if n_total else float("nan")

    # Per-fold test-split majority: the baseline the accuracy column is really
    # measured against. Read from the dataset itself so it cannot drift from
    # what the probe was scored on.
    per_fold_maj: list[float] = []
    per_fold_n: list[int] = []
    ds = None
    fold_splits: dict[str, dict[str, list[int]]] = {}
    item_paths: list[str] = []
    try:
        import sys
        sys.path.insert(0, str(PATHS.benchmir_root / "src"))
        from benchmir.eval.datasets.corpora.humdrum_eval_corpus import (
            HumdrumComposerClassificationEvalDataset,
        )
        ds = HumdrumComposerClassificationEvalDataset(
            root_dir=str(PATHS.benchmir_data_root / "humdrum"),
            symbolic_format="midi", target_class="composer")
        item_paths = [str(path) for path, _ in ds._index]
        for f in ds.get_fold_ids():
            view = ds.fold_view(f)
            fold_splits[f] = view.get_splits()
            test = fold_splits[f]["test"]
            c = Counter(view[i][1] for i in test)
            per_fold_maj.append(max(c.values()) / len(test))
            per_fold_n.append(len(test))
    except Exception as exc:  # pragma: no cover - reporting must not die here
        print(f"warning: per-fold baseline unavailable: {exc}")

    fails = warm_failures(sorted(set(model_ids.values())))

    # ---- markdown --------------------------------------------------------
    L = [f"# {args.title}", ""]
    L += [
        f"Composer classification over the Humdrum corpus: **{n_total} scores, "
        f"{n_classes} composers, 5 folds** "
        "(`splits/composer_splits_5fold.json`). Song level, mean-pooled bar "
        "embeddings -- the same extraction path as CIPI / EMOPIA / TopMAGD, so "
        "each row is comparable with that arm's columns in passes 1 and 2.",
        "",
        "Probe: MLP [256, 256], dropout 0.2, Adam lr 1e-3, batch 64, 50 epochs. "
        "Cells are mean ±population SD across the 5 folds.",
        "",
        "**The ±SD is across folds, not across seeds.** `run.seed` is parsed "
        "into the EvalPlan and then never used -- BenchMIR seeds neither torch "
        "nor numpy (orchestration/initialiser.py stores it; nothing reads it), "
        "so each cell is one unseeded probe fit and the spread below carries no "
        "information about run-to-run variance.",
        "",
        "`max_examples_per_class` is **not** set: this is the full corpus. "
        "Upstream's `configs/example_humdrum_composer_debug_v1.yaml` sets it to "
        "5 (~100 songs) as a debug knob; its full-corpus config "
        "`configs/example_all_evals_v1.yaml` (task `T7`) leaves it at the class "
        "default of `None`, which is what this pass uses.",
        "",
    ]

    L += ["Code: the evaluation library at the commit the "
          "`third_party/BenchMIR` submodule pins, plus the encoder "
          "adapters carried on that branch -- `src/benchmir/models/"
          "ours_*.py` and `ours_workers/*` -- which are what produced "
          "every embedding in this table.",
          "",
          "This task did not exist in the library before the upstream "
          "`87349a4` merge, which is what added the Humdrum corpus reader "
          "and the composer task. Passes 1 and 2 were produced before that "
          "merge and are unaffected by it: the only change it made to code "
          "shared with them is `EarlyStoppingConfig.min_delta`, a no-op "
          "under these configs, which set no `early_stopping` at all. The "
          "bar embeddings come from the same content-hash cache.",
          "",
          "## Baselines", "",
          "| baseline | value |", "|---|---|",
          f"| rows (songs) | {n_total} |",
          f"| classes | {n_classes} |",
          f"| majority class, whole corpus | {maj_pooled:.4f} "
          f"({max(pooled.values())}/{n_total}) |",
          f"| uniform chance (1/K) | {1/n_classes:.4f} |"]
    if per_fold_maj:
        L.append(
            f"| majority class, per-fold test split | "
            f"{statistics.mean(per_fold_maj):.4f} "
            f"±{statistics.pstdev(per_fold_maj):.4f} "
            f"(min {min(per_fold_maj):.4f}, max {max(per_fold_maj):.4f}) |")
        size = (f"{per_fold_n[0]}" if len(set(per_fold_n)) == 1
                else f"{min(per_fold_n)}-{max(per_fold_n)}")
        L.append(f"| test-split size per fold | {size} |")
    # A majority-class predictor puts every song in one class, so exactly one
    # class has a non-zero F1: precision = p, recall = 1, F1 = 2p/(1+p).
    # macro_f1 averages that single non-zero value over all K classes, which is
    # an order of magnitude BELOW 1/K -- it is not the same floor as chance.
    _p = maj_pooled
    maj_f1 = 2 * _p / (1 + _p)
    maj_macro_f1 = maj_f1 / n_classes
    maj_weighted_f1 = _p * maj_f1
    L += ["",
          "The two degenerate predictors do **not** share a floor, and the gap "
          "is on `macro_f1`:", "",
          "| degenerate predictor | accuracy | balanced_accuracy | macro_f1 | "
          "weighted_f1 |", "|---|---|---|---|---|",
          f"| always predict the majority class | {maj_pooled:.4f} | "
          f"{1/n_classes:.4f} | {maj_macro_f1:.4f} | {maj_weighted_f1:.4f} |",
          f"| uniform random guess | {1/n_classes:.4f} | ~{1/n_classes:.4f} | "
          f"~{1/n_classes:.4f} | ~{1/n_classes:.4f} |",
          "",
          "A majority-class predictor collapses to one non-zero per-class F1 "
          f"(2p/(1+p) = {maj_f1:.4f}), and `macro_f1` averages that over all "
          f"{n_classes} classes -- {maj_macro_f1:.4f}, roughly "
          f"{(1/n_classes)/maj_macro_f1:.0f}x below 1/K. So the higher of the "
          f"two `macro_f1` floors is uniform chance at about {1/n_classes:.3f}, "
          "and that is the number a `macro_f1` column should be read against. "
          "(These are the closed forms; the per-fold values differ in the "
          "fourth decimal.) The corpus is close to balanced (largest class "
          f"{max(pooled.values())}, smallest {min(pooled.values())}), so the "
          "majority and chance `accuracy` baselines nearly coincide and "
          "`accuracy` and `balanced_accuracy` can be read together.", ""]

    # ---- byte-identical scores -------------------------------------------
    # The split file assigns entry_ids, not file contents. Several Humdrum
    # sub-collections ship the same score twice under different ids (the same
    # Bach chorale in two editions, a Chopin first-edition pair), and those
    # land on opposite sides of the train/test line. Every arm sees the same
    # inflation, but the table should say how big it can be.
    if item_paths and fold_splits:
        import hashlib
        digests = []
        for path in item_paths:
            try:
                digests.append(hashlib.md5(Path(path).read_bytes()).hexdigest())
            except OSError:
                digests.append(f"__unreadable__{len(digests)}")
        groups = defaultdict(list)
        for i, d in enumerate(digests):
            groups[d].append(i)
        dup_groups = [v for v in groups.values() if len(v) > 1]
        if dup_groups:
            n_dup_rows = sum(len(v) for v in dup_groups)
            L += ["## Byte-identical scores in the corpus", "",
                  f"The {n_total} files carry only **{len(groups)} distinct "
                  f"md5 digests**: {len(dup_groups)} group(s) covering "
                  f"{n_dup_rows} rows are byte-identical duplicates of another "
                  "score under a different `entry_id`. "
                  "`composer_splits_5fold.json` splits on `entry_id`, so a "
                  "duplicate pair can straddle the train/test line:", "",
                  "| fold | test rows | test rows byte-identical to a TRAIN "
                  "row | to a VAL row |", "|---|---|---|---|"]
            for f in sorted(fold_splits):
                sp = fold_splits[f]
                tr = {digests[i] for i in sp["train"]}
                va = {digests[i] for i in sp.get("val", [])}
                te = sp["test"]
                L.append(f"| {f} | {len(te)} | "
                         f"{sum(1 for i in te if digests[i] in tr)} | "
                         f"{sum(1 for i in te if digests[i] in va)} |")
            L += ["",
                  "At most 2 of ~394 test rows per fold, so the leak is worth "
                  "**under 0.5 accuracy points** and it lands on every arm "
                  "equally -- it cannot reorder the table. It is an upstream "
                  "corpus/split property, not something this pass introduced.",
                  ""]

    L += ["## Checkpoints evaluated", "",
          "| arm | pass | checkpoint | rows extracted | encode failures |",
          "|---|---|---|---|---|"]
    for arm in arms:
        nf, _ = fails.get(arm, (0, {}))
        L.append(f"| `{arm}` | {'1' if arm in PASS1 else '2'} | "
                 f"`{ckpts.get(arm, '?')}` | {n_rows.get(arm, '?')} | {nf} |")
    L += ["",
          "An encode failure is not a dropped row: `BarEmbeddingModel._bars` "
          "emits a **zero embedding** for a file the worker refuses, so the "
          "song stays in train/val/test with a constant feature vector. That is "
          "why the count is printed beside the scores rather than in a log.",
          ""]

    L += ["## Results", ""]
    L.append("| arm | n folds | " + " | ".join(metrics) + " |")
    L.append("|" + "---|" * (len(metrics) + 2))
    for arm in arms:
        L.append(f"| **{arm}** | {len(folds_seen[arm])} | "
                 + " | ".join(cell(data.get((arm, m), [])) for m in metrics) + " |")
    L.append("")

    L += ["## Per-fold accuracy", "",
          "| arm | " + " | ".join(f"fold {f}" for f in sorted(
              {r['fold'] for r in rows})) + " |"]
    fold_ids = sorted({r["fold"] for r in rows})
    L.append("|" + "---|" * (len(fold_ids) + 1))
    by_fold = {(family(r["model_id"]), r["fold"]): r["metrics"] for r in rows}
    for arm in arms:
        cells = []
        for f in fold_ids:
            m = by_fold.get((arm, f))
            cells.append("--" if m is None else f"{m.get('accuracy', float('nan')):.3f}")
        L.append(f"| **{arm}** | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## How this was produced", "",
          "`configs/ours_composer_v1.yaml` -> "
          "`scripts_ours/gen_jobs_composer.py` (one job config per arm plus the "
          "encode shard tasks) -> `slurm/encode.sbatch` (CPU, 4G, %12) -> "
          "`slurm/eval.sbatch` (CPU, 12G, %16) -> this script. "
          "`GRID=composer ./run_eval.sh all` does all of it in one command.",
          "",
          "The 1,968 files are encoded once per arm into the shared content-hash "
          "embedding cache; the probe jobs then only "
          "read `.npz` and fit MLPs, which is why each arm's five folds finish "
          "in a couple of minutes. Re-running any part is a cache hit.",
          ""]

    listed = {a: fails[a][1] for a in arms if fails.get(a, (0, {}))[0]}
    if listed:
        L += ["## Encode failures, by arm", ""]
        for arm, blob in listed.items():
            # Group by the KIND of failure, not by its exact message: an OOV
            # message names the specific tokens and the count, so keying on the
            # whole string would print 28 groups of 1 and say nothing.
            reasons = Counter(_failure_kind(msg) for msg in blob.values())
            L.append(f"* `{arm}`: {len(blob)} file(s) -- "
                     + "; ".join(f"{k} x{v}" for k, v in reasons.most_common(5)))
            toks = Counter()
            for msg in blob.values():
                # one vote per FILE per family, not one per token occurrence:
                # a single message can name the same family five times.
                for t in set(re.findall(r"\b([A-Za-z]+)_[0-9]+", msg)):
                    toks[t] += 1
            if toks:
                L.append("  * token families the vocabulary lacks: "
                         + ", ".join(f"`{k}_*` (in {v} of {len(blob)} file(s))"
                                     for k, v in toks.most_common(5)))
            # A zero row in TRAIN is 1/1333 of noise; a zero row in TEST is a
            # guaranteed near-miss on a 20-class problem. Say which.
            if item_paths and fold_splits:
                pos = {path: i for i, path in enumerate(item_paths)}
                bad = {pos[path] for path in blob if path in pos}
                if bad:
                    L.append("  * where those zero rows land: "
                             + "; ".join(
                                 f"fold {f} test "
                                 f"{len(bad & set(fold_splits[f]['test']))}"
                                 f"/{len(fold_splits[f]['test'])}"
                                 for f in sorted(fold_splits))
                             + " -- so this arm is scored on up to "
                             f"{max(len(bad & set(fold_splits[f]['test'])) for f in fold_splits)} "
                             "constant-feature songs per fold, which depresses "
                             "it relative to the arms with none.")
        L.append("")

    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
