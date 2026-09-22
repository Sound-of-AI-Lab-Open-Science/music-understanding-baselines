#!/bin/bash
# Pre-train one arm.
#
#   ./run_pretrain.sh jepa_paper            # the paper-true Music-JEPA recipe
#   ./run_pretrain.sh jepa_champA           # the smaller collapse control
#   ./run_pretrain.sh musetok
#   ./run_pretrain.sh musicbert
#   ./run_pretrain.sh jepa_paper --smoke    # a handful of files, a few steps
#
# The submitted job SELF-CHAINS: it queues its own successor before training, so
# a partition walltime cap cannot end a run without a queued continuation, and
# the chain stops itself when checkpoints/final.ckpt appears. JR_CHAIN=0 submits
# a single job. A smoke is never chained.
#
# --smoke merges the recipe's own `smoke:` block over the recipe -- production
# batch size, sequence length and precision, only the step count and the split
# wiring change -- so a smoke's throughput extrapolates and the smoke of an arm
# cannot drift from the arm.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh
jr_prepare_dirs

ARM="${1:?usage: ./run_pretrain.sh <arm> [--smoke] [train.py args...]   arms: $JR_ARMS}"
shift || true
jr_arm_model "$ARM" >/dev/null || exit 2

SMOKE=0
for a in "$@"; do [ "$a" = "--smoke" ] && SMOKE=1; done
if [ "$SMOKE" = "1" ]; then
  TIME="${JR_PRETRAIN_SMOKE_TIME:-01:00:00}"; MEM="${JR_PRETRAIN_SMOKE_MEM:-48G}"
  export JR_CHAIN=0
else
  TIME="${JR_PRETRAIN_TIME:-1-00:00:00}"; MEM="${JR_PRETRAIN_MEM:-64G}"
fi

cd "$WORK_ROOT"
# shellcheck disable=SC2046
JOB=$(sbatch --parsable $(jr_sbatch_args --gpu) \
        --output="$LOG_ROOT/%x_%j.log" \
        --job-name="jr_pre_${ARM}" --time="$TIME" --mem="$MEM" \
        --cpus-per-task="${JR_PRETRAIN_CPUS:-4}" \
        "$PKG_ROOT/slurm/pretrain.sbatch" "$ARM" "$@")
echo "pretrain $ARM: job $JOB"
echo "logs: $LOG_ROOT/jr_pre_${ARM}_${JOB}.log"
if [ "$SMOKE" = "1" ]; then
  echo "next: ./run_snapshot.sh $ARM latest --smoke   (the run is $RUN_ROOT/${ARM}_smoke)"
else
  echo "next: ./run_snapshot.sh $ARM latest && ./run_eval.sh <eval-arm>"
fi
