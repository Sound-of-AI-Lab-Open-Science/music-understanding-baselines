#!/bin/bash
#SBATCH --job-name=jr_encode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=01:50:00
#SBATCH --output=logs/%x_%A_%a.log
# ^ last-resort default only. Slurm resolves it against the SUBMISSION
# directory, so it works when submitted from $WORK_ROOT (what the run_*.sh
# runners do) and nowhere else. Every runner overrides it with an absolute
# --output under $LOG_ROOT; pass the same flag if you submit this by hand.
#
# Fill the per-bar embedding cache for one model over one shard of the
# evaluation corpora.  CPU by default; pass JR_ENCODE_DEVICE=cuda and submit
# with a GPU to use one.
#
#   sbatch --output="$LOG_ROOT/%x_%A_%a.log" --array=1-12%12 slurm/encode.sbatch
#
# `benchmir run` would extract these embeddings itself, but it commits its own
# cache only once a whole (dataset, model) extraction finishes -- so a job
# killed at the walltime limit loses everything.  warm_cache.py drives the same
# encoders through the same content-hash cache one file at a time, so a killed
# shard loses at most the file it was on and a resubmit picks up from there.
#
# It reads $JOBS_DIR/_resolved_config.yaml -- the config with the checkpoint
# overrides ALREADY APPLIED -- and never the recipe in configs/.  Pointing it at
# the recipe warms whatever checkpoint that file still names: the cache then
# reports thousands of hits in a minute and warms nothing, and the probe jobs
# re-encode the whole corpus inline.  The generated resolved config is what
# makes a warm shard and an eval job provably the same weights.
#
# 4G is measured, not guessed (MaxRSS 0.0-1.9 GB over 36 shards, MuseTok at the
# top of the range).  Memory is what caps array concurrency, so measuring it is
# what makes %12 affordable.
#
# One line of _warm_tasks.txt per array index: "<model_id> <shard_i> <shard_n>".
set -uo pipefail
: "${PKG_ROOT:?source env.sh before submitting (it exports PKG_ROOT)}"
source "$PKG_ROOT/env.sh"

JOBS_DIR="${JR_JOBS_DIR:?JR_JOBS_DIR must name the generated jobs directory}"
DEVICE="${JR_ENCODE_DEVICE:-cpu}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

read -r MID SI SN < <(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$JOBS_DIR/_warm_tasks.txt")
[ -n "${MID:-}" ] || { echo "no task at line ${SLURM_ARRAY_TASK_ID:-1}"; exit 0; }

PY="$(jr_python "$BENCHMIR_ENV")" || exit 1
echo "[encode] $MID shard $SI/$SN device=$DEVICE start=$(date -Is)"
"$PY" -u "$PKG_ROOT/eval/benchmir/scripts_ours/warm_cache.py" \
  --config "$JOBS_DIR/_resolved_config.yaml" \
  --model-id "$MID" --device "$DEVICE" --shard "${SI}/${SN}" \
  ${JR_ENCODE_DATASETS:+$JR_ENCODE_DATASETS}
RC=$?
echo "[encode] $MID shard $SI/$SN rc=$RC end=$(date -Is)"
exit $RC
