#!/bin/bash
# Verify that the corpora are where the recipes expect them, BEFORE a grid is
# submitted. Every path it prints is derived from $BENCHMIR_DATA_ROOT and
# $MIDI_DIR, so relocating the data means exporting one variable, not editing
# anything.
#
#   ./check_data.sh
#
# None of these corpora is redistributed with this package; each is obtained
# from its own source under its own terms. See THIRD_PARTY_NOTICES.md.
#
# Exit status is the number of missing entries, so this can gate a pipeline.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh

MISSING=0
have () {                       # $1 = path, $2 = what it is
  if [ -e "$1" ]; then
    printf '  ok       %-58s %s\n' "${1#"$BENCHMIR_DATA_ROOT"/}" "$2"
  else
    printf '  MISSING  %-58s %s\n' "${1#"$BENCHMIR_DATA_ROOT"/}" "$2"
    MISSING=$((MISSING + 1))
  fi
}
glob_have () {                  # $1 = directory, $2 = pattern, $3 = what it is
  local n
  n=$(find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then
    printf '  ok       %-58s %s (%d)\n' "${1#"$BENCHMIR_DATA_ROOT"/}/$2" "$3" "$n"
  else
    printf '  MISSING  %-58s %s\n' "${1#"$BENCHMIR_DATA_ROOT"/}/$2" "$3"
    MISSING=$((MISSING + 1))
  fi
}

echo "pre-training corpus   MIDI_DIR=$MIDI_DIR"
if [ -d "$MIDI_DIR" ]; then
  N=$(find "$MIDI_DIR" -maxdepth 2 -name '*.mid*' 2>/dev/null | head -1000 | wc -l)
  printf '  %-8s %s\n' "$([ "$N" -gt 0 ] && echo ok || echo EMPTY)" \
         "$N MIDI files found in the first two levels (search is recursive)"
  [ "$N" -eq 0 ] && MISSING=$((MISSING + 1))
else
  echo "  MISSING  $MIDI_DIR  (set DATA_ROOT, or MIDI_DIR directly)"
  MISSING=$((MISSING + 1))
fi

echo
echo "evaluation corpora    BENCHMIR_DATA_ROOT=$BENCHMIR_DATA_ROOT"
have "$BENCHMIR_DATA_ROOT/cipi/difficulty_cipi"          "CIPI piano difficulty (Henle grades)"
have "$BENCHMIR_DATA_ROOT/emopia/EMOPIA_2.2"             "EMOPIA emotion (4Q)"
have "$BENCHMIR_DATA_ROOT/humdrum/composer_humdrum"      "Humdrum/KernScores composer classification"
have "$BENCHMIR_DATA_ROOT/LMDMatched/lmd_matched"        "Lakh MIDI, matched subset"
glob_have "$BENCHMIR_DATA_ROOT/LMDMatched" '*.cls'       "TopMAGD genre + partition files"
have "$BENCHMIR_DATA_ROOT/POP909cl/POP909_processed"     "POP909 audio-aligned MIDI"
have "$BENCHMIR_DATA_ROOT/POP909cl/POP909_chord_annotated" "POP909 chord annotations"

echo
echo "checkpoints           CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
# None of these three is produced by this package: all are externally supplied
# weights for the `v1` reference grid, and $JEPA_CKPT_SEED2 is its second-seed
# JEPA control. Override any of them by exporting it. See THIRD_PARTY_NOTICES.md.
for f in "$JEPA_CKPT" "$JEPA_CKPT_SEED2" "$MUSICBERT_CKPT"; do
  if [ -e "$f" ]; then printf '  ok       %s\n' "$f"
  else printf '  absent   %s  (supply it yourself; only the reference grid needs it)\n' "$f"; fi
done
for ARM in $JR_ARMS; do
  f="$CHECKPOINT_ROOT/$ARM.ckpt"
  if [ -e "$f" ]; then printf '  ok       %-46s -> %s\n' "$ARM.ckpt" "$(readlink -f "$f")"
  else printf '  absent   %-46s (run_snapshot.sh writes it)\n' "$ARM.ckpt"; fi
done

echo
if [ "$MISSING" -eq 0 ]; then
  echo "all required corpora present"
else
  echo "$MISSING missing entr$([ "$MISSING" -eq 1 ] && echo y || echo ies)"
fi
exit "$MISSING"
