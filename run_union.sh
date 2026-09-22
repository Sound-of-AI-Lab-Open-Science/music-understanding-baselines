#!/bin/bash
# Build the split, the union caches and the MuseTok vocabulary from the shard
# caches that run_tokenize.sh produced. Runs in the foreground: it is minutes of
# hashing and symlinking, not a training job.
#
#   ./run_union.sh                       # both codecs, no corpus restriction
#   ./run_union.sh --keep-ids ids.txt    # restrict to a keep-list (see below)
#   SPLIT_TAG=filtered ./run_union.sh --keep-ids ids.txt
#
# Order matters and is not negotiable:
#
#   1. SPLIT ON CONTENT, NOT ON ID. --content-hash-from keys the 98/1/1 split on
#      the file's content hash, which the tokenization indexes already carry.
#      Keying it on the corpus id instead leaked 12.1% of the held-out test set
#      into train, because the same piece is uploaded under many ids. The script
#      must report "0 ids with no content hash"; anything else means the Octuple
#      tokenization is incomplete and the split would silently leak.
#
#   2. UNION THE SHARD CACHES, dropping the held-out test ids entirely -- a
#      test song must not be reachable from the training corpus at all. The
#      merge writes symlinks, so this costs minutes and no disk.
#
#   3. DERIVE THE MUSETOK VOCABULARY FROM THE CORPUS. Upstream's released
#      168-token dictionary comes from a piano corpus; on a broad corpus ~24% of
#      pieces carry at least one event it cannot express, and the dataset raises
#      KeyError on the first one. Deriving it is also what the paper did.
#      CONSEQUENCE: token ids shift, so checkpoints trained here are NOT
#      vocabulary-compatible with the public MuseTok weights.
#
# A corpus restriction is applied HERE, as a keep-list on the already-built
# caches -- never as `data.filter` in a training config, which would change the
# tokenization cache fingerprint and force a full re-tokenization. Give a
# restricted run its own SPLIT_TAG: the merge WRITES pieces_{train,val}.txt into
# the split directory, so two variants sharing one directory overwrite each
# other's piece lists.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh
jr_prepare_dirs

KEEP=""
CODECS="octuple remi"
while [ $# -gt 0 ]; do
  case "$1" in
    --keep-ids) KEEP="${2:?--keep-ids needs a file}"; shift 2 ;;
    --codec)    CODECS="${2:?--codec needs octuple or remi}"; shift 2 ;;
    *) echo "usage: ./run_union.sh [--keep-ids FILE] [--codec octuple|remi]" >&2; exit 2 ;;
  esac
done
[ -n "$KEEP" ] && jr_require "$KEEP" "keep-list"

PY_J="$(jr_python "$JEPA_ENV")"
PY_M="$(jr_python "$MUSETOK_ENV")"
cd "$PKG_ROOT/pretrain"

echo "=== 1/3 split (content-hash) -> $SPLIT_DIR"
"$PY_J" scripts/make_musescore_split.py \
    --corpus "$MIDI_DIR" --out "$SPLIT_DIR" \
    --content-hash-from "$CACHE_ROOT" \
    ${KEEP:+--keep-ids "$KEEP"}

for CODEC in $CODECS; do
  OUT="$(jr_union_cache "$CODEC")"
  echo "=== 2/3 union $CODEC -> $OUT"
  PY="$PY_J"; [ "$CODEC" = "remi" ] && PY="$PY_M"
  "$PY" baselines/cache_tools/merge_shard_caches.py --codec "$CODEC" \
      --cache-root "$CACHE_ROOT" --out "$OUT" --split-dir "$SPLIT_DIR" \
      ${KEEP:+--keep-ids "$KEEP"}
done

case " $CODECS " in
  *" remi "*)
    echo "=== 3/3 MuseTok vocabulary -> $VOCAB_ROOT/dictionary_musescore.pkl"
    "$PY_M" baselines/cache_tools/build_musescore_vocab.py \
        --cache-root "$CACHE_ROOT" \
        --out "$VOCAB_ROOT/dictionary_musescore.pkl" --workers 16
    ;;
  *) echo "=== 3/3 skipped (no REMI+ codec requested)" ;;
esac

echo
echo "done. next: ./run_pretrain.sh <arm> [--smoke]   (arms: $JR_ARMS)"
