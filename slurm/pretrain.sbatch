#!/bin/bash
#SBATCH --job-name=jr_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.log
# ^ last-resort default only. Slurm resolves it against the SUBMISSION
# directory, so it works when submitted from $WORK_ROOT (what the run_*.sh
# runners do) and nowhere else. Every runner overrides it with an absolute
# --output under $LOG_ROOT; pass the same flag if you submit this by hand.
#
# Pre-train one arm.  Submitted by run_pretrain.sh; the #SBATCH values above are
# defaults it overrides with the cluster flags from env.sh.
#
#   sbatch --output="$LOG_ROOT/%x_%j.log" slurm/pretrain.sbatch <arm> [--smoke] [extra train.py args...]
#
# SELF-CHAINING.  Before training starts this job queues its own successor with
# --dependency=afterany against the same --out-dir and --resume, so a partition
# walltime cap can never end a run without a queued continuation.  The chain
# stops itself: the successor sees checkpoints/final.ckpt and exits without
# training.  Set JR_CHAIN=0 to submit a single un-chained job.
#
# The arm is a POSITIONAL argument, never an environment variable, because a
# cluster that sets SBATCH_EXPORT=NONE would drop the variable on the chained
# resubmit and the successor would silently train a different arm into the same
# directory.  The chain re-executes SLURM's spool copy of this file, so editing
# it cannot affect a running chain -- but the YAML is re-read from the package
# at every segment, so a config edit DOES reach the next segment.  Do not edit a
# recipe while its chain is alive.
set -uo pipefail
: "${PKG_ROOT:?source env.sh before submitting (it exports PKG_ROOT)}"
source "$PKG_ROOT/env.sh"

ARM="${1:?usage: sbatch slurm/pretrain.sbatch <arm> [--smoke] [train.py args...]}"
shift || true
MODEL="$(jr_arm_model "$ARM")" || exit 2
CODEC="$(jr_arm_codec "$ARM")" || exit 2
ENVNAME="$(jr_arm_env "$ARM")" || exit 2
CACHE="$(jr_union_cache "$CODEC")"
PY="$(jr_python "$ENVNAME")" || exit 1

SMOKE=0
for a in "$@"; do [ "$a" = "--smoke" ] && SMOKE=1; done
if [ "$SMOKE" = "1" ]; then
  OUT="$RUN_ROOT/${ARM}_smoke"
else
  OUT="$RUN_ROOT/$ARM"
fi

jr_require "$CACHE" "run run_union.sh first" || exit 1

if [ -f "$OUT/checkpoints/final.ckpt" ]; then
  echo "[chain] $OUT/checkpoints/final.ckpt exists -- run finished, nothing to do."
  exit 0
fi

if [ "${JR_CHAIN:-1}" = "1" ] && [ "$SMOKE" = "0" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  # The successor is submitted with the same shape as this job. The defaults
  # here are the ones run_pretrain.sh uses, so a chain keeps its allocation even
  # on a cluster that sets SBATCH_EXPORT=NONE and drops the variables.
  # shellcheck disable=SC2046
  NEXT=$(JR_CHAIN=1 sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" \
           $(jr_sbatch_args --gpu) \
           --output="$LOG_ROOT/%x_%j.log" \
           --job-name="jr_pre_${ARM}" \
           --time="${JR_PRETRAIN_TIME:-1-00:00:00}" \
           --mem="${JR_PRETRAIN_MEM:-64G}" \
           --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
           "$0" "$ARM" "$@" 2>&1) \
    && echo "[chain] successor queued: $NEXT" \
    || echo "[chain] WARNING could not queue successor: $NEXT"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
command -v nvidia-smi >/dev/null && \
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[pretrain] arm=$ARM model=$MODEL codec=$CODEC out=$OUT start=$(date -Is)"
cd "$PKG_ROOT/pretrain" || exit 1
"$PY" baselines/train.py \
    --model "$MODEL" \
    --config "baselines/configs/${ARM}.yaml" \
    --cache-prebuilt "$CACHE" \
    --out-dir "$OUT" \
    --resume "$@"
RC=$?
echo "[pretrain] arm=$ARM rc=$RC end=$(date -Is)"
exit $RC
