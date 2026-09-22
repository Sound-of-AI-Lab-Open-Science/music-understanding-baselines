#!/bin/bash
# Evaluate one arm: generate its job configs, fill the embedding cache, then fit
# the probes.
#
#   ./run_eval.sh jepa_musescore_paper
#   ./run_eval.sh musetok_musescore cipi_difficulty emopia_emotion
#   GRID=composer ./run_eval.sh all
#   GRID=v1 ./run_eval.sh all           # the released-checkpoint reference grid
#
# LAUNCH ONE ARM AT A TIME, the moment that arm stops training. Waiting for the
# whole grid puts the slowest run on every other run's critical path. Each arm
# writes into its own jobs directory and a shared runs root, and
# ./run_report.sh merges whatever has finished -- it can be re-run later
# with more columns.
#
# The probe array depends on the encode array with `afterany`, not `afterok`: a
# shard killed at the wall has still written every file it finished (the cache
# is per file), and a probe job falls back to encoding a miss inline. `afterok`
# would throw away a 95%-warm cache over one bad shard.
#
# Nothing here writes into a training run. Reading a frozen checkpoint snapshot
# is the only contact with one.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh
source ./lib/grids.sh
jr_prepare_dirs

ARM="${1:?usage: ./run_eval.sh <arm|all> [task ...]   GRID=$JR_GRIDS}"
shift || true

WARM=$(./run_encode.sh "$ARM" "$@" | tee /dev/stderr | sed -n 's/^WARM_JOBID=//p')
jr_jobs_dir_for "$ARM"                 # re-derive $JR_JOBS_DIR in this shell
DEP=""
[ -n "${WARM:-}" ] && DEP="--dependency=afterany:$WARM"

C="${JR_EVAL_CONCURRENCY:-20}"
export JR_JOBS_DIR
cd "$WORK_ROOT"

submit_list () {                        # $1 = list file, $2 = memory, $3 = label
  local list="$1" mem="$2" label="$3" n
  n=$(wc -l < "$JR_JOBS_DIR/$list" 2>/dev/null || echo 0)
  [ "$n" -eq 0 ] && return 0
  local id
  # shellcheck disable=SC2046
  id=$(JR_JOB_LIST="$list" sbatch --parsable $(jr_sbatch_args) \
         --output="$LOG_ROOT/%x_%A_%a.log" \
         --job-name="jr_eval_${GRID}" --array="1-${n}%${C}" --mem="$mem" \
         --time="${JR_EVAL_TIME:-03:45:00}" $DEP \
         "$PKG_ROOT/slurm/eval.sbatch")
  echo "eval $label: $id ($n jobs, %$C, $mem)"
}

if [ -f "$JR_JOBS_DIR/_jobs_light.txt" ]; then
  # The split is by MEASURED memory, not by task; see slurm/eval.sbatch.
  submit_list _jobs_light.txt "${JR_EVAL_MEM:-12G}"       "light"
  submit_list _jobs_heavy.txt "${JR_EVAL_HEAVY_MEM:-16G}" "heavy"
else
  submit_list _all_jobs.txt   "${JR_EVAL_MEM:-12G}"       "all"
fi

echo
echo "watch: squeue -u \"\$USER\""
echo "then:  ./run_report.sh ${GRID}"
