"""Split configs/ours_composer_v1.yaml into one SLURM job per arm.

Same reasoning as scripts_ours/gen_jobs_pass2.py: BenchMIR's probes are torch
MLPs on a CPU-only torch, and each job needs its own `output_dir` because
Worker hardcodes its embedding cache to `output_dir/cache/embeddings` and
writes `report.json` there.

Simpler than pass 2 in one way and wider in another:

* one task, so the unit of work is just the arm -- and all five folds stay
  inside one job, because Worker extracts embeddings ONCE per (model, dataset)
  and reuses them across folds (`extractor.extract(base_dataset,
  split_name="all")`, orchestration/worker.py). Splitting by fold would make
  four of the five folds re-read the same 1,968-row blob for nothing;
* eight arms, not four: pass 1's four released/previously-trained checkpoints
  and pass 2's four MuseScore retrains, side by side in one table.

An arm whose checkpoint is not on disk is DROPPED with a printed line rather
than generating a job that dies at model construction.
"""

from __future__ import annotations

import argparse
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

SRC = PATHS.pkg_root / "eval" / "benchmir" / "configs" / "ours_composer_v1.yaml"
JOBS = PATHS.jobs_root / "composer"
RUNS = PATHS.run_root / "composer"

#: Table order: pass 1 first, then pass 2, each in the order they were trained.
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


#: A pass-1 arm names no checkpoint_path -- it takes its adapter's DEFAULT_CKPT.
#: Recorded here explicitly so `_checkpoints.txt`, and therefore the table,
#: names the actual weights rather than "<class default>". A table that cannot
#: say which checkpoint produced a number is not a table.
CLASS_DEFAULTS = {
    "MusicBERTEmbeddingModel":
        str(PATHS.musicbert_ckpt),
    "MuseTokEmbeddingModel":
        str(PATHS.musetok_ckpt),
    "MusicJEPAEmbeddingModel":
        str(PATHS.jepa_ckpt),
}


def checkpoint_of(entry: dict) -> str:
    return entry["kwargs"].get("checkpoint_path") or CLASS_DEFAULTS[entry["class"]]


def family(model_id: str) -> str:
    return model_id[: -len("-song")] if model_id.endswith("-song") else model_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-runs", default=str(RUNS))
    ap.add_argument("--jobs-dir", default=str(JOBS))
    ap.add_argument("--arms", action="append", default=[],
                    help="restrict to these arms (repeatable)")
    args = ap.parse_args()

    runs = Path(args.out_runs)
    jobs = Path(args.jobs_dir)

    base = expandvars_tree(yaml.safe_load(SRC.read_text()))
    models_by_arm = {family(m["model_id"]): dict(m) for m in base["model"]}
    unknown = sorted(set(models_by_arm) - set(ARMS))
    if unknown:
        raise SystemExit(f"config has arms not in ARMS: {unknown}")
    ds_by_id = {d["dataset_id"]: d for d in base["datasets"]}

    wanted = set(args.arms) if args.arms else None
    if wanted:
        bad = sorted(wanted - set(ARMS))
        if bad:
            raise SystemExit(f"unknown arm(s) {bad}; choose from {ARMS}")

    arms: list[str] = []
    for arm in ARMS:
        if arm not in models_by_arm:
            continue
        if wanted and arm not in wanted:
            continue
        ck = checkpoint_of(models_by_arm[arm])
        if not Path(ck).is_file():
            print(f"SKIPPING arm {arm}: no checkpoint at {ck}")
            continue
        arms.append(arm)
    if not arms:
        raise SystemExit("no arm has a checkpoint; nothing to run")

    if jobs.exists():
        shutil.rmtree(jobs)
    jobs.mkdir(parents=True)

    task = base["eval_tasks"][0]
    written: list[str] = []
    for arm in arms:
        entry = models_by_arm[arm]
        job_id = f"{task['task_id']}__{arm}"
        t = dict(task)
        t["model_ids"] = [entry["model_id"]]
        cfg = {
            "run": {
                "run_id": job_id,
                "seed": base["run"]["seed"],
                "output_dir": str(runs / job_id),
            },
            "model": [entry],
            "datasets": [ds_by_id[task["dataset_id"]]],
            "eval_tasks": [t],
            "report": base["report"],
        }
        (jobs / f"{job_id}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        written.append(job_id)

    # The warm array reads THIS file, never the repo config, for the same
    # reason gen_jobs_pass2.py writes one: a warm shard and a probe job that
    # disagree about which weights they are for warm one cache while the probe
    # re-encodes into another, silently. Here the two are the same file, but
    # keeping the indirection means a future --ckpt override cannot reintroduce
    # the split.
    resolved = dict(base)
    resolved["model"] = [models_by_arm[a] for a in arms]
    (jobs / "_resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False))

    write_lines(jobs / "_all_jobs.txt", written)

    n_shards = int(os.environ.get("BENCHMIR_COMPOSER_WARM_SHARDS", "12"))
    warm = [f"{models_by_arm[a]['model_id']} {i} {n_shards}"
            for a in arms for i in range(n_shards)]
    write_lines(jobs / "_warm_tasks.txt", warm)

    lines = [f"{a}\t{checkpoint_of(models_by_arm[a])}" for a in arms]
    write_lines(jobs / "_checkpoints.txt", lines)

    print(f"{len(written)} job configs and {len(warm)} warm shards in {jobs}")
    print(f"output_dir root: {runs}")
    for line in lines:
        print("  ", line)


if __name__ == "__main__":
    main()
