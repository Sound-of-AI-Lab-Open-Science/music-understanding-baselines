#!/bin/bash
# Freeze a training checkpoint into $CHECKPOINT_ROOT, where the evaluation
# reads it.
#
#   ./run_snapshot.sh jepa_paper latest
#   ./run_snapshot.sh musetok final
#   ./run_snapshot.sh musicbert /path/to/musicbert-periodic-step=018000.ckpt
#   ./run_snapshot.sh jepa_paper latest --smoke   # from runs/<arm>_smoke
#
# --smoke reads the SMOKE run's directory instead of the real one, because
# run_pretrain.sh <arm> --smoke writes to $RUN_ROOT/<arm>_smoke and this script
# would otherwise look only at $RUN_ROOT/<arm> and report "no checkpoint" for a
# smoke that just succeeded. The flag is explicit, and the stamped copy is named
# <arm>_smoke_step<N>.ckpt, precisely so that 30 steps of proof cannot be
# mistaken on disk for a finished run -- the whole point of freezing a snapshot
# is that a table can say which weights produced it.
#
# Why copy at all: the training jobs are live, so a checkpoint can be half
# written under a reader, and an evaluation that reads a moving file is not
# reproducible. The copy is also the record of WHICH weights a pass evaluated.
#
# Two names are written, on purpose:
#   <arm>_step<N>.ckpt   the immutable, stamped copy -- this is what the job
#                        configs record, so a finished pass says on disk which
#                        step produced it
#   <arm>.ckpt           a symlink to the newest stamped copy -- what the
#                        evaluation recipes name, so they never need editing
#
# A spec is `final` (written at max_steps), `latest` (the highest-numbered
# periodic checkpoint, else last.ckpt) or a path.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh
jr_prepare_dirs

ARM=""; SPEC=""; SMOKE=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    *) if [ -z "$ARM" ]; then ARM="$a"; elif [ -z "$SPEC" ]; then SPEC="$a"; else
         echo "unexpected argument '$a'" >&2; exit 2; fi ;;
  esac
done
[ -n "$ARM" ] || { echo "usage: ./run_snapshot.sh <arm> [final|latest|/path/to.ckpt] [--smoke]   arms: $JR_ARMS" >&2; exit 2; }
SPEC="${SPEC:-latest}"
jr_arm_model "$ARM" >/dev/null || exit 2

if [ "$SMOKE" = "1" ]; then RUN_NAME="${ARM}_smoke"; TAG="_smoke"; else RUN_NAME="$ARM"; TAG=""; fi
CKPTS="$RUN_ROOT/$RUN_NAME/checkpoints"
case "$SPEC" in
  final)  SRC="$CKPTS/final.ckpt" ;;
  latest) # A glob, not `ls | sort | tail`: under `set -o pipefail` that pipeline
          # returns ls's status 2 when the glob matches nothing, `set -e` kills
          # the script on the assignment, and the last.ckpt fallback on the next
          # line -- which exists exactly for a run that has not reached its first
          # periodic checkpoint -- is unreachable. The periodic filenames are
          # zero-padded ({step:06d}), so the shell's lexicographic glob order is
          # numeric order and `sort` bought nothing.
          SRC=""
          for f in "$CKPTS"/*periodic-step=*.ckpt; do [ -f "$f" ] && SRC="$f"; done
          [ -n "$SRC" ] || SRC="$CKPTS/last.ckpt" ;;
  *)      SRC="$SPEC" ;;
esac
[ -f "$SRC" ] || { echo "no checkpoint for $ARM: $SRC" >&2; exit 1; }

STEP=$(basename "$SRC" | sed -n 's/.*step=\([0-9]*\)\.ckpt/\1/p')
[ -n "$STEP" ] || STEP=$(basename "$SRC" .ckpt)
DST="$CHECKPOINT_ROOT/${ARM}${TAG}_step${STEP}.ckpt"

# write-then-rename: an evaluation job must never see a half-copied checkpoint
cp -f "$SRC" "$DST.tmp.$$"
mv -f "$DST.tmp.$$" "$DST"
ln -sfn "$(basename "$DST")" "$CHECKPOINT_ROOT/${ARM}.ckpt"

echo "snapshot: $DST"
echo "stable:   $CHECKPOINT_ROOT/${ARM}.ckpt -> $(basename "$DST")"
[ "$SMOKE" = "1" ] && echo "NOTE: this is a SMOKE checkpoint. Anything evaluated against it is a
      proof that the wiring runs, not a result."
exit 0
