"""One SLURM job per (split strategy, arm) for TopMAGD genre classification.

Three settings x eight arms = 24 jobs:

  kfold5    stratified_kfold, n_folds=5     -- the setting every existing
            TopMAGD number in runs/v1 and runs/pass2 was computed
            under. Re-run here as a CONTROL, because the merge of upstream
            87349a4 changed how the class weights are derived (train split
            instead of whole corpus), so "new strategy vs old number" would
            otherwise confound the split change with a loss change.
  balanced  balanced_fixed_size             -- TU Wien's fixed 2,000-track-per-
            genre TRAIN quota (msd-topMAGD-partition_fixedSizeSplit_2000-v1.0)
  stratpct  stratified_percentage_split     -- TU Wien's fixed 80/20 per genre
            (msd-topMAGD-partition_stratifiedPercentageSplit_0.8-v1.0)

The model entry for each arm is copied VERBATIM out of that arm's existing
topmagd job config, so the checkpoint path, cache_name and every other kwarg
are identical to what produced the 5-fold column. Nothing here re-derives them.

`use_all_matches: false` on every job, which is what all eight existing
TopMAGD runs used -- upstream only changed the class DEFAULT from True to
False, and our configs always set it explicitly, so the song set is unchanged
(10,282 tracks) across all three columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# The package's one path configuration (published_code/pkgpaths.py). It reads
# paths.yaml and the same environment variables env.sh exports, so a default
# printed by --help is the path this script will actually use.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pkgpaths import PATHS, write_lines  # noqa: E402

PASS1_JOBS = PATHS.jobs_root / "v1"
PASS2_JOBS = PATHS.jobs_root / "pass2"
ROOT_DIR = str(PATHS.benchmir_data_root / "LMDMatched") + "/"

#: arm -> the existing topmagd job config to lift the `model:` entry from.
ARM_SOURCES = {
    # pass 1: released / previously trained checkpoints
    "musicbert":            PASS1_JOBS / "topmagd_genre__musicbert.yaml",
    "musetok":              PASS1_JOBS / "topmagd_genre__musetok.yaml",
    "music-jepa":           PASS1_JOBS / "topmagd_genre__music-jepa.yaml",
    "music-jepa-champA":    PASS1_JOBS / "topmagd_genre__music-jepa-champA.yaml",
    # pass 2: MuseScore-retrained checkpoints
    "musicbert_musescore":  PASS2_JOBS / "musicbert_musescore/topmagd_genre__musicbert_musescore.yaml",
    "musetok_musescore":    PASS2_JOBS / "musetok_musescore/topmagd_genre__musetok_musescore.yaml",
    "jepa_musescore_champA": PASS2_JOBS / "jepa_musescore_champA/topmagd_genre__jepa_musescore_champA.yaml",
    "jepa_musescore_paper": PASS2_JOBS / "jepa_musescore_paper/topmagd_genre__jepa_musescore_paper.yaml",
}

SETTINGS = {
    "kfold5":   {"split_strategy": "stratified_kfold", "n_folds": 5},
    "balanced": {"split_strategy": "balanced_fixed_size"},
    "stratpct": {"split_strategy": "stratified_percentage_split"},
}

PROBE = {"type": "mlp", "hidden_dims": [256, 256], "dropout": 0.2}
TRAINING = {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 64,
    "max_epochs": 50,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", default=str(PATHS.jobs_root / "topmagd_splits"))
    ap.add_argument("--out-runs", default=str(PATHS.run_root / "topmagd_splits"))
    args = ap.parse_args()

    jobs_dir = Path(args.jobs_dir)
    runs = Path(args.out_runs)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    for arm, src in ARM_SOURCES.items():
        if not src.exists():
            print(f"SKIP {arm}: no source config at {src}")
            continue
        blob = yaml.safe_load(src.read_text())
        model_entries = blob["model"]
        if len(model_entries) != 1:
            raise SystemExit(f"{src} has {len(model_entries)} model entries, expected 1")
        model_id = model_entries[0]["model_id"]

        for setting, ds_kwargs in SETTINGS.items():
            job = f"topmagd_{setting}__{arm}"
            cfg = {
                "run": {
                    "run_id": job,
                    "seed": 42,
                    "output_dir": str(runs / job),
                },
                "model": model_entries,
                "datasets": [
                    {
                        "dataset_id": "topmagd",
                        "class": "MSDTopMAGDGenreClassificationEvalDataset",
                        "root_dir": ROOT_DIR,
                        "kwargs": {
                            "use_all_matches": False,
                            "val_ratio": 0.15,
                            **ds_kwargs,
                        },
                    }
                ],
                "eval_tasks": [
                    {
                        "task_id": "topmagd_genre",
                        "class": "MSDTopMAGDGenreClassificationDownstreamEvalTask",
                        "dataset_id": "topmagd",
                        "model_ids": [model_id],
                        "folds": "all",
                        "metrics": ["accuracy"],
                        "probe": PROBE,
                        "training": TRAINING,
                    }
                ],
                "report": {"format": "json"},
            }
            (jobs_dir / f"{job}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            names.append(job)

    # `_all_jobs.txt` and not `_jobs.txt`: every grid in this package writes
    # the same file names, so one SLURM template drives all of them.
    write_lines(jobs_dir / "_all_jobs.txt", names)
    print(f"wrote {len(names)} job configs to {jobs_dir}")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
