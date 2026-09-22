#!/bin/bash
# Generate one grid's job configs and fill the per-bar embedding cache for it.
#
#   ./run_encode.sh jepa_musescore_paper
#   GRID=composer ./run_encode.sh all
#
# run_eval.sh calls this and then submits the probes, so you rarely run it on
# its own -- do that when you want the expensive half (encoding) to finish and
# be inspected before any probe is scheduled.
#
# Prints `WARM_JOBID=<id>` as its last line so a caller can depend on it.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh
source ./lib/grids.sh
jr_prepare_dirs

ARM="${1:?usage: ./run_encode.sh <arm|all> [task ...]   GRID=$JR_GRIDS}"
shift || true

jr_generate_jobs "$ARM" "$@"
NW=$(wc -l < "$JR_JOBS_DIR/_warm_tasks.txt" 2>/dev/null || echo 0)
if [ "$NW" -eq 0 ]; then
  echo "nothing to encode for grid '$GRID' (no _warm_tasks.txt)"
  echo "WARM_JOBID="
  exit 0
fi

C="${JR_ENCODE_CONCURRENCY:-12}"
export JR_JOBS_DIR
cd "$WORK_ROOT"
# shellcheck disable=SC2046
JOB=$(sbatch --parsable $(jr_sbatch_args) \
        --output="$LOG_ROOT/%x_%A_%a.log" \
        --job-name="jr_enc_${GRID}" --array="1-${NW}%${C}" \
        --mem="${JR_ENCODE_MEM:-4G}" --time="${JR_ENCODE_TIME:-01:50:00}" \
        "$PKG_ROOT/slurm/encode.sbatch")
echo "encode $GRID/$ARM: job $JOB ($NW shards, %$C, ${JR_ENCODE_MEM:-4G})"
echo "WARM_JOBID=$JOB"
