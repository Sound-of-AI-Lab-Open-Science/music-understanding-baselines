"""Split configs/ours_all_evals_v1.yaml into one SLURM job per unit of work.

Why split at all: BenchMIR's probes are torch MLPs and this env's torch is
CPU-only, because no CUDA wheel exists that also satisfies the project's
`torch>=2.13` pin against many drivers (max CUDA 12.8; cu128 tops out
at torch 2.11). Measured on POP909-chord fold 0, one probe fit is ~474 s on 8
CPU threads, so the whole 6-task x 3-model x folds grid is ~14 h serially --
far past a typical short-queue walltime.

Splitting turns that into ~30 independent jobs that each finish well inside the
limit. The unit is (task, model), except POP909-key, whose ~62k training frames
push a single fold to ~45 min -- that one is split per fold as well.

Each job gets its own output_dir because Worker hardcodes the embedding cache
to `output_dir/cache/embeddings` and writes `report.json` there; sharing one
directory would make concurrent jobs race on both. Re-extraction is cheap: the
expensive per-bar embeddings are already in the content-hash cache, so an
extraction is just npz reads.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

# The package's one path configuration (published_code/pkgpaths.py). It reads
# paths.yaml and the same environment variables env.sh exports, so a default
# printed by --help is the path this script will actually use.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pkgpaths import PATHS, expandvars_tree, write_lines  # noqa: E402

SRC = PATHS.pkg_root / "eval" / "benchmir" / "configs" / "ours_all_evals_v1.yaml"
JOBS = PATHS.jobs_root / "v1"
RUNS = PATHS.run_root / "v1"

# task_id -> split per fold? (only the one that needs it)
PER_FOLD = {"pop909_key"}
MODELS = ["music-jepa", "musicbert", "musetok", "music-jepa-champA"]


def main() -> None:
    base = expandvars_tree(yaml.safe_load(SRC.read_text()))
    if JOBS.exists():
        shutil.rmtree(JOBS)
    JOBS.mkdir(parents=True)

    models_by_id = {m["model_id"]: m for m in base["model"]}
    ds_by_id = {d["dataset_id"]: d for d in base["datasets"]}

    written: list[str] = []
    for task in base["eval_tasks"]:
        for family in MODELS:
            mid = next(m for m in task["model_ids"] if m.startswith(family + "-"))
            folds = task.get("folds")
            if task["task_id"] in PER_FOLD and folds == "all":
                # fold ids come from the dataset; POP909 uses n_folds=5 -> "0".."4"
                fold_sets = [[i] for i in range(ds_by_id[task["dataset_id"]]["kwargs"]["n_folds"])]
            else:
                fold_sets = [folds]

            for fs in fold_sets:
                suffix = "" if fs in (None, "all") else f"_f{fs[0]}"
                job_id = f"{task['task_id']}__{family}{suffix}"
                t = dict(task)
                t["model_ids"] = [mid]
                if fs is None:
                    t.pop("folds", None)
                else:
                    t["folds"] = fs
                cfg = {
                    "run": {
                        "run_id": job_id,
                        "seed": base["run"]["seed"],
                        "output_dir": str(RUNS / job_id),
                    },
                    "model": [models_by_id[mid]],
                    "datasets": [ds_by_id[task["dataset_id"]]],
                    "eval_tasks": [t],
                    "report": base["report"],
                }
                path = JOBS / f"{job_id}.yaml"
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                written.append(job_id)

    write_lines(JOBS / "_all_jobs.txt", written)
    warm_ids = [f"{family}-song" for family in MODELS]
    # Every grid writes the same three files into its jobs directory, so one
    # SLURM template can drive all of them:
    #   _resolved_config.yaml  the config with every override already applied --
    #                          what the encode array must read, never the recipe
    #   _warm_tasks.txt        one "<model_id> <shard_i> <shard_n>" per line
    #   _all_jobs.txt          one generated job config name per line
    (JOBS / "_resolved_config.yaml").write_text(
        yaml.safe_dump(base, sort_keys=False))
    n_shards = int(os.environ.get("JR_ENCODE_SHARDS", "12"))
    warm = [f"{mid} {i} {n_shards}"
            for mid in warm_ids for i in range(n_shards)]
    write_lines(JOBS / "_warm_tasks.txt", warm)
    print(f"{len(written)} job configs and {len(warm)} warm shards in {JOBS}")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
