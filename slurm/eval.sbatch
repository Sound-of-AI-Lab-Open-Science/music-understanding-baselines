#!/bin/bash
#SBATCH --job-name=jr_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=03:45:00
#SBATCH --output=logs/%x_%A_%a.log
# ^ last-resort default only. Slurm resolves it against the SUBMISSION
# directory, so it works when submitted from $WORK_ROOT (what the run_*.sh
# runners do) and nowhere else. Every runner overrides it with an absolute
# --output under $LOG_ROOT; pass the same flag if you submit this by hand.
#
# Run one generated evaluation job: extract (or read from cache) the embeddings
# for one (task, arm) unit and fit its probe.  CPU only -- by the time this runs
# the embeddings are already in the content-hash cache, so the job reads npz
# files and fits MLPs.  A stray cache miss falls back to encoding inside the
# worker rather than failing the job.
#
#   JR_JOBS_DIR=... JR_JOB_LIST=_jobs_light.txt \
#     sbatch --output="$LOG_ROOT/%x_%A_%a.log" --array=1-40%20 slurm/eval.sbatch
#
# The job list is split by MEASURED memory, not by task: over one 40-job grid
# the 30-song POP909 folds ran 0.42-1.51 GB MaxRSS while the 909-song folds ran
# 0.99-4.02 GB.  One request sized for the heavy half was ~30x what the light
# half needed, and the request -- not the work -- was what capped concurrency.
# run_eval.sh therefore submits _jobs_light.txt at 12G and _jobs_heavy.txt at
# 16G as two arrays.
set -uo pipefail
: "${PKG_ROOT:?source env.sh before submitting (it exports PKG_ROOT)}"
source "$PKG_ROOT/env.sh"

JOBS_DIR="${JR_JOBS_DIR:?JR_JOBS_DIR must name the generated jobs directory}"
LIST="${JR_JOB_LIST:-_all_jobs.txt}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

JOB=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$JOBS_DIR/$LIST")
[ -n "$JOB" ] || { echo "no job at line ${SLURM_ARRAY_TASK_ID:-1} of $LIST"; exit 0; }

PY="$(jr_python "$BENCHMIR_ENV")" || exit 1
echo "[eval] job=$JOB start=$(date -Is)"
"$PY" -u -m benchmir.cli.main run --config "$JOBS_DIR/${JOB}.yaml"
RC=$?
echo "[eval] job=$JOB rc=$RC end=$(date -Is)"
exit $RC
