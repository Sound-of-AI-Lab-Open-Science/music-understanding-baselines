"""Per-fold SLURM job configs for POP909 chord/root on ALL 909 songs.

Same probe protocol and same checkpoints as ours_v1 -- the only difference is
`max_songs: all` on the dataset, which lifts the historical [:30] slice (see
pop909cl_eval_corpus.resolve_max_songs). ours_v1's own configs are untouched,
so its 30-song numbers stay reproducible.

Split per fold because the frame count goes from ~18k to ~550k rows: one fold's
probe fit is hours, not minutes, and a single (task, model) job would be five of
those back to back.
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
JOBS = PATHS.jobs_root / "pop909full"
RUNS = PATHS.run_root / "pop909full"

TASKS = ["pop909_chord", "pop909_root"]
MODELS = ["music-jepa", "musicbert", "musetok", "music-jepa-champA"]
N_FOLDS = 5


def main() -> None:
    base = expandvars_tree(yaml.safe_load(SRC.read_text()))
    if JOBS.exists():
        shutil.rmtree(JOBS)
    JOBS.mkdir(parents=True)

    models_by_id = {m["model_id"]: m for m in base["model"]}
    ds_by_id = {d["dataset_id"]: d for d in base["datasets"]}

    written: list[str] = []
    for task in base["eval_tasks"]:
        if task["task_id"] not in TASKS:
            continue
        ds = dict(ds_by_id[task["dataset_id"]])
        ds["kwargs"] = dict(ds["kwargs"], max_songs="all")
        for family in MODELS:
            mid = next(m for m in task["model_ids"] if m.startswith(family + "-"))
            for fold in range(N_FOLDS):
                job_id = f"{task['task_id']}__{family}_f{fold}"
                t = dict(task)
                t["model_ids"] = [mid]
                t["folds"] = [fold]
                cfg = {
                    "run": {
                        "run_id": job_id,
                        "seed": base["run"]["seed"],
                        "output_dir": str(RUNS / job_id),
                    },
                    "model": [models_by_id[mid]],
                    "datasets": [ds],
                    "eval_tasks": [t],
                    "report": base["report"],
                }
                (JOBS / f"{job_id}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
                written.append(job_id)

    write_lines(JOBS / "_all_jobs.txt", written)
    warm_ids = [f"{family}-frame8" for family in MODELS]
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


if __name__ == "__main__":
    main()
