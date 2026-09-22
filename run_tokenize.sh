#!/bin/bash
# Tokenize the pre-training corpus, one SLURM array task per shard.
#
#   ./run_tokenize.sh octuple          # Music-JEPA and MusicBERT share this cache
#   ./run_tokenize.sh remi             # MuseTok
#   ./run_tokenize.sh octuple 5,7,12   # re-run three shards that failed
#
# A "shard" is an immediate subdirectory of $MIDI_DIR; a flat corpus is one
# shard. Tokenization is cached per (codec, exact file list), so re-running a
# finished shard costs nothing and a killed one resumes.
#
# Concurrency is capped with %N because tokenization is I/O bound on shared
# storage: more tasks than the filesystem can feed makes every task slower.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh
jr_prepare_dirs

CODEC="${1:?usage: ./run_tokenize.sh <octuple|remi> [array-spec]}"
case "$CODEC" in octuple|remi) ;; *) echo "codec must be octuple or remi" >&2; exit 2 ;; esac

jr_require "$MIDI_DIR" "set DATA_ROOT or MIDI_DIR to the pre-training corpus" || exit 1

N=$(find "$MIDI_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$N" -eq 0 ] && N=1
CONCURRENCY="${JR_TOKENIZE_CONCURRENCY:-5}"
ARRAY="${2:-0-$((N - 1))%${CONCURRENCY}}"

# REMI+ is an order of magnitude slower per file than OctupleMIDI, so it gets a
# longer default walltime. Both are resumable, so a short cap is survivable.
if [ "$CODEC" = "remi" ]; then TIME="${JR_TOKENIZE_TIME:-13:00:00}"
else                          TIME="${JR_TOKENIZE_TIME:-01:50:00}"; fi

cd "$WORK_ROOT"
# shellcheck disable=SC2046
JOB=$(sbatch --parsable $(jr_sbatch_args) \
        --output="$LOG_ROOT/%x_%A_%a.log" \
        --job-name="jr_tok_${CODEC}" --array="$ARRAY" --time="$TIME" \
        --mem="${JR_TOKENIZE_MEM:-24G}" \
        "$PKG_ROOT/slurm/tokenize.sbatch" "$CODEC")
echo "tokenize $CODEC: job $JOB (array $ARRAY, $N shards)"
echo "logs: $LOG_ROOT/jr_tok_${CODEC}_${JOB}_*.log"
echo "next: ./run_union.sh"
