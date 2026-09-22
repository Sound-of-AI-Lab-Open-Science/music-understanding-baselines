#!/bin/bash
#SBATCH --job-name=jr_tokenize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=13:00:00
#SBATCH --output=logs/%x_%A_%a.log
# ^ last-resort default only. Slurm resolves it against the SUBMISSION
# directory, so it works when submitted from $WORK_ROOT (what the run_*.sh
# runners do) and nowhere else. Every runner overrides it with an absolute
# --output under $LOG_ROOT; pass the same flag if you submit this by hand.
#
# Tokenize one shard of the pre-training corpus.  CPU only -- no GPU is used or
# requested.  Submitted by run_tokenize.sh, which sizes the array and passes the
# cluster flags; the #SBATCH values above are only the defaults it overrides.
#
#   sbatch --output="$LOG_ROOT/%x_%A_%a.log" --array=0-19%5 slurm/tokenize.sbatch octuple
#   sbatch --output="$LOG_ROOT/%x_%A_%a.log" --array=0-19%10 slurm/tokenize.sbatch remi
#
# One array task per immediate subdirectory of $MIDI_DIR, sorted, so the task
# id -> shard mapping is stable across resubmits and a failed shard can be
# resubmitted on its own (--array=7).  A corpus with no subdirectories is one
# shard: array task 0 tokenizes the whole tree.
#
# Rates measured on a 1.7M-file corpus with 8 workers: OctupleMIDI ~70 files/s
# (~19 min for an 85k-file shard), REMI+ 4-13 files/s (~6 h).  That is why REMI+
# gets the long default walltime and why both are resumable: REMI+ has a
# per-file SIGALRM and skips pickles that already exist, so a task killed at the
# wall can simply be resubmitted for the same shard.
set -uo pipefail
: "${PKG_ROOT:?source env.sh before submitting (it exports PKG_ROOT)}"
source "$PKG_ROOT/env.sh"

CODEC="${1:?usage: sbatch slurm/tokenize.sbatch <octuple|remi> [corpus-dir]}"
CORPUS="${2:-$MIDI_DIR}"
case "$CODEC" in
  octuple) MODEL=musicbert; ENVNAME="$JEPA_ENV"   ; CONFIG=musicbert.yaml ;;
  remi)    MODEL=musetok  ; ENVNAME="$MUSETOK_ENV"; CONFIG=musetok.yaml   ;;
  *) echo "codec must be octuple or remi, got '$CODEC'" >&2; exit 2 ;;
esac

# The shard list, recomputed identically in every task.
mapfile -t SHARDS < <(find "$CORPUS" -mindepth 1 -maxdepth 1 -type d | sort)
[ ${#SHARDS[@]} -eq 0 ] && SHARDS=("$CORPUS")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#SHARDS[@]}" ]; then
  echo "[tokenize] array index $IDX >= ${#SHARDS[@]} shards; nothing to do"
  exit 0
fi
SHARD="${SHARDS[$IDX]}"

PY="$(jr_python "$ENVNAME")" || exit 1
echo "[tokenize] codec=$CODEC shard=$IDX dir=$SHARD start=$(date -Is)"
cd "$PKG_ROOT/pretrain" || exit 1
"$PY" baselines/train.py --model "$MODEL" \
    --config "baselines/configs/$CONFIG" \
    --midi-dir "$SHARD" \
    --cache-dir "$CACHE_ROOT" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" --tokenize-only
RC=$?
echo "[tokenize] codec=$CODEC shard=$IDX rc=$RC end=$(date -Is)"
exit $RC
