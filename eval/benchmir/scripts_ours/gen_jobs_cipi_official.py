"""Per-(arm, fold) SLURM job configs for CIPI on the official movement-level tree.

20 jobs: 4 pass-1 arms x 5 folds. Split per fold like everything else in
scripts_ours, so a job is minutes rather than one job being five probe fits back
to back, and so a single fold can be resubmitted on its own.

Also writes `_warm_tasks.txt`: 4 arms x 4 shards over the 592 movements. The
warm step is separate from the eval step because five folds of one arm would
otherwise each try to encode the same 592 files at the same time -- correct
(the cache write is write-then-rename) but four times wasted.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

# The package's one path configuration (published_code/pkgpaths.py). It reads
# paths.yaml and the same environment variables env.sh exports, so a default
# printed by --help is the path this script will actually use.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pkgpaths import PATHS, expandvars_tree, write_lines  # noqa: E402

SRC = PATHS.pkg_root / "eval" / "benchmir" / "configs" / "ours_cipi_official_v1.yaml"
JOBS = PATHS.jobs_root / "cipi_official"
RUNS = PATHS.run_root / "cipi_official"
N_FOLDS = 5
WARM_SHARDS = 4
ARMS = ["music-jepa", "music-jepa-champA", "musicbert", "musetok"]


def main() -> None:
    base = expandvars_tree(yaml.safe_load(SRC.read_text()))
    if JOBS.exists():
        shutil.rmtree(JOBS)
    JOBS.mkdir(parents=True)

    models_by_id = {m["model_id"]: m for m in base["model"]}
    ds = base["datasets"][0]
    task = base["eval_tasks"][0]

    written: list[str] = []
    for arm in ARMS:
        mid = f"{arm}-song"
        for fold in range(N_FOLDS):
            job_id = f"cipi_difficulty__{arm}_f{fold}"
            t = dict(task, model_ids=[mid], folds=[fold])
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
    warm = [f"{arm}-song {i} {WARM_SHARDS}" for arm in ARMS for i in range(WARM_SHARDS)]
    write_lines(JOBS / "_warm_tasks.txt", warm)
    # The encode array must read the config with every override applied, never
    # the recipe in configs/ -- see slurm/encode.sbatch.
    (JOBS / "_resolved_config.yaml").write_text(
        yaml.safe_dump(base, sort_keys=False))
    print(f"{len(written)} job configs and {len(warm)} warm shards in {JOBS}")


if __name__ == "__main__":
    main()
