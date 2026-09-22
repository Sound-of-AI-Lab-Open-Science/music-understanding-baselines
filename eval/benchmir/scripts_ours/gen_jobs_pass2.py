"""Split configs/ours_all_evals_pass2.yaml into one SLURM job per unit of work.

Same reasoning as scripts_ours/gen_jobs.py: BenchMIR's probes are torch MLPs on
a CPU-only torch, so the whole grid run serially is many hours, and each job
needs its own output_dir because Worker hardcodes its embedding cache to
`output_dir/cache/embeddings` and writes report.json there.

What is different from pass 1
-----------------------------
* four arms instead of three-plus-a-control, three of them MuseScore-retrained;
* POP909 chord/root appear twice, on the 30-song slice and on all 909;
* the three retrained checkpoint paths can be overridden per invocation
  (--ckpt jepa_musescore_paper=/path/to/final.ckpt, or the environment
  variables BENCHMIR_PASS2_CKPT_<ARM>), so run_eval.sh can point the grid at a
  finished run without editing the config. The override is applied to the
  generated job configs, and every job config records the path it used -- a
  pass whose jobs disagree about which checkpoint they evaluated is not a pass.

The unit of work is (task, arm), except for the three tasks whose row counts
make one fold a job on its own: pop909_key (~62k frames/fold) and the two
909-song POP909 tasks (~550k rows instead of ~18k).
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

SRC = PATHS.pkg_root / "eval" / "benchmir" / "configs" / "ours_all_evals_pass2.yaml"
JOBS = PATHS.jobs_root / "pass2"
RUNS = PATHS.run_root / "pass2"

#: One job per fold for these -- a single (task, arm) job would be five long
#: probe fits back to back.
PER_FOLD = {"pop909_key", "pop909_chord_full", "pop909_root_full"}

#: Tasks whose jobs need the 16G sbatch rather than the 12G one. Measured on
#: pass 1: its 909-song POP909 folds ran 0.99-4.02 GB MaxRSS (mean 2.64) while
#: every other job in the grid ran 0.42-1.51 GB (mean 0.64). Splitting the array
#: by that fact is what lets the light half go to %20 -- one 48G request for
#: both was ~30x the light jobs' need and the request, not the work, was the
#: thing capping concurrency.
HEAVY_TASKS = {"pop909_chord_full", "pop909_root_full"}

ARMS = [
    "musicbert",
    "jepa_musescore_paper",
    "jepa_musescore_champA",
    "musetok_musescore",
    "musicbert_musescore",
]
#: The arms whose checkpoint may be swapped. The pass-1 `musicbert` reference is
#: not one of them: it is the fixed point the table is read against, and swapping
#: it would make every other column incomparable with pass 1.
SWAPPABLE = ARMS[1:]


def resolve_overrides(pairs: list[str]) -> dict[str, str]:
    over: dict[str, str] = {}
    for arm in SWAPPABLE:
        env = os.environ.get("BENCHMIR_PASS2_CKPT_" + arm.upper())
        if env:
            over[arm] = env
    for pair in pairs or []:
        arm, _, path = pair.partition("=")
        if arm not in SWAPPABLE:
            raise SystemExit(f"--ckpt arm must be one of {SWAPPABLE}, got {arm!r}")
        if not path:
            raise SystemExit(f"--ckpt needs arm=path, got {pair!r}")
        over[arm] = path
    for arm, path in over.items():
        if not Path(path).is_file():
            raise SystemExit(f"checkpoint for {arm} does not exist: {path}")
    return over


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[],
                    help="arm=/path/to/checkpoint.ckpt (repeatable)")
    ap.add_argument("--out-runs", default=str(RUNS),
                    help="output_dir root for the generated jobs")
    ap.add_argument("--jobs-dir", default=str(JOBS))
    ap.add_argument("--only", action="append", default=[],
                    help="restrict to these task_ids (repeatable)")
    ap.add_argument("--arms", action="append", default=[],
                    help="restrict to these arms (repeatable). Single-arm mode: "
                         "each retrained model can be evaluated the moment it "
                         "finishes training, instead of waiting for the slowest "
                         "one -- which is the whole grid's critical path when "
                         "the runs finish hours apart.")
    args = ap.parse_args()

    over = resolve_overrides(args.ckpt)
    runs = Path(args.out_runs)
    jobs = Path(args.jobs_dir)

    base = expandvars_tree(yaml.safe_load(SRC.read_text()))
    models_by_id = {m["model_id"]: dict(m) for m in base["model"]}
    for mid, m in models_by_id.items():
        arm = mid.rsplit("-", 1)[0]
        if arm in over:
            m["kwargs"] = dict(m["kwargs"], checkpoint_path=over[arm])
    ds_by_id = {d["dataset_id"]: d for d in base["datasets"]}

    # An arm whose checkpoint is not on disk is DROPPED, loudly, rather than
    # generating jobs that die at model construction. The MuseScore MusicBERT
    # run is the reason: the grid has to be runnable before that run has
    # produced a checkpoint, and a missing arm is a missing column in the
    # table -- which is honest -- while a crashed array is 16 red jobs and no
    # table at all.
    wanted = set(args.arms) if args.arms else None
    if wanted:
        unknown = sorted(wanted - set(ARMS))
        if unknown:
            raise SystemExit(f"unknown arm(s) {unknown}; choose from {ARMS}")
    arms = []
    for arm in ARMS:
        if wanted and arm not in wanted:
            continue
        ck = models_by_id[f"{arm}-song"]["kwargs"].get("checkpoint_path")
        if ck and not Path(ck).is_file():
            print(f"SKIPPING arm {arm}: no checkpoint at {ck}")
            continue
        arms.append(arm)
    if not arms:
        raise SystemExit("no arm has a checkpoint; nothing to run")

    warm_ids = [f"{a}-song" for a in SWAPPABLE if a in arms]

    if jobs.exists():
        # keep _warm_tasks.txt: it is rewritten below and nothing else reads it
        shutil.rmtree(jobs)
    jobs.mkdir(parents=True)

    written: list[str] = []
    light: list[str] = []
    heavy: list[str] = []
    for task in base["eval_tasks"]:
        if args.only and task["task_id"] not in args.only:
            continue
        for arm in arms:
            mid = next(m for m in task["model_ids"] if m.rsplit("-", 1)[0] == arm)
            folds = task.get("folds")
            if task["task_id"] in PER_FOLD and folds == "all":
                n = ds_by_id[task["dataset_id"]]["kwargs"]["n_folds"]
                fold_sets = [[i] for i in range(n)]
            else:
                fold_sets = [folds]

            for fs in fold_sets:
                suffix = "" if fs in (None, "all") else f"_f{fs[0]}"
                job_id = f"{task['task_id']}__{arm}{suffix}"
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
                        "output_dir": str(runs / job_id),
                    },
                    "model": [models_by_id[mid]],
                    "datasets": [ds_by_id[task["dataset_id"]]],
                    "eval_tasks": [t],
                    "report": base["report"],
                }
                (jobs / f"{job_id}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
                written.append(job_id)
                (heavy if task["task_id"] in HEAVY_TASKS else light).append(job_id)

    # _all_jobs.txt is kept for the single-array launch path; _jobs_light.txt
    # and _jobs_heavy.txt are what the two sbatch scripts actually read, because
    # the two halves want different memory.
    # The warm array reads the config through warm_cache.py --config, and it must
    # be the config with the checkpoint overrides ALREADY APPLIED. Pointing it at
    # the repo's ours_all_evals_pass2.yaml warms whatever checkpoint that file's
    # anchor still names, which on 2026-09-09 was a stale snapshot: the warm
    # array reported 12,755 cache hits in 82 seconds while the eval jobs, reading
    # the overridden per-job YAMLs, found a cold cache and re-encoded 10,282
    # TopMAGD files inline. Same failure the --config/--model-id design was meant
    # to remove, one level up. Write the resolved config next to the jobs.
    resolved = dict(base)
    resolved["model"] = [models_by_id[m["model_id"]] for m in base["model"]]
    (jobs / "_resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False))

    write_lines(jobs / "_all_jobs.txt", written)
    write_lines(jobs / "_jobs_light.txt", light)
    write_lines(jobs / "_jobs_heavy.txt", heavy)

    # The warm-cache task list is written from the SAME resolved model entries,
    # so a warm shard and an eval job can never be pointed at different weights.
    n_shards = int(os.environ.get("BENCHMIR_PASS2_WARM_SHARDS", "12"))
    warm = [f"{mid} {i} {n_shards}" for mid in warm_ids for i in range(n_shards)]
    write_lines(jobs / "_warm_tasks.txt", warm)

    # ...and the checkpoint each arm resolved to, next to them, so a finished
    # pass says on disk which weights produced it.
    lines = []
    for arm in arms:
        kw = models_by_id[f"{arm}-song"]["kwargs"]
        lines.append(f"{arm}\t{kw.get('checkpoint_path', '<class default>')}")
    write_lines(jobs / "_checkpoints.txt", lines)

    print(f"{len(written)} job configs ({len(light)} light / {len(heavy)} heavy) "
          f"and {len(warm)} warm shards in {jobs}")
    print(f"output_dir root: {runs}")
    for line in lines:
        print("  ", line)


if __name__ == "__main__":
    main()
