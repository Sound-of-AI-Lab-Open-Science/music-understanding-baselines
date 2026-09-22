# The evaluation grids, and the one function that turns (grid, arm) into a
# generated jobs directory. Sourced by run_encode.sh and run_eval.sh.
#
# A "grid" is one evaluation campaign: which recipe, which generator, where its
# jobs and results live. They exist separately because they answer different
# questions and were run at different times, and because merging them would make
# numbers that are not comparable look comparable.
#
#   pass2          the main grid: the arms this package pre-trains, on every task
#   v1             the same tasks against released / previously trained weights,
#                  the reference column every other number is read against
#   composer       composer classification, all arms of both grids
#   cipi_official  difficulty on the official movement-level CIPI index
#   pop909full     POP909 chord/root on all 909 songs rather than the 30-song
#                  historical slice -- never compare the two with each other
#   topmagd        genre under three split strategies; it LIFTS each arm's model
#                  entry out of the pass2/v1 job configs, so run those first
#
# Comments and generated reports throughout the package say "pass 1" and
# "pass 2". Those are the same two grids: pass 1 == `v1`, pass 2 == `pass2`.
# They are called passes because they were run in that order, `v1` first.
#
# shellcheck shell=bash

JR_GRIDS="pass2|v1|composer|cipi_official|pop909full|topmagd"
: "${GRID:=pass2}"
export GRID JR_GRIDS

jr_jobs_dir_for() {              # $1 = arm -> sets $JR_JOBS_DIR
  case "$GRID" in
    pass2)         JR_JOBS_DIR="$JOBS_ROOT/pass2/$1" ;;
    composer)      JR_JOBS_DIR="$JOBS_ROOT/composer" ;;
    v1)            JR_JOBS_DIR="$JOBS_ROOT/v1" ;;
    cipi_official) JR_JOBS_DIR="$JOBS_ROOT/cipi_official" ;;
    pop909full)    JR_JOBS_DIR="$JOBS_ROOT/pop909full" ;;
    topmagd)       JR_JOBS_DIR="$JOBS_ROOT/topmagd_splits" ;;
    *) echo "unknown GRID '$GRID' (have: $JR_GRIDS)" >&2; return 2 ;;
  esac
  export JR_JOBS_DIR
}

#: Training arm -> the name the EVALUATION grids call the same model. The two
#: namespaces differ because an eval arm names a (model, pre-training corpus)
#: pair -- `musicbert` is the released reference and `musicbert_musescore` is
#: the one this package trains -- while a training arm names a recipe. One
#: function so that the pairing cannot drift between the checkpoint overrides
#: below and anything else that needs it (run_smoke.sh --end-to-end does).
jr_eval_arm() {
  case "$1" in
    jepa_paper)  printf 'jepa_musescore_paper' ;;
    jepa_champA) printf 'jepa_musescore_champA' ;;
    musetok)     printf 'musetok_musescore' ;;
    musicbert)   printf 'musicbert_musescore' ;;
    *) echo "unknown arm '$1' (have: $JR_ARMS)" >&2; return 2 ;;
  esac
}

jr_generate_jobs() {             # $1 = arm, $2.. = task ids to restrict to
  local arm="$1"; shift || true
  local py only=() armargs=()
  py="$(jr_python "$BENCHMIR_ENV")" || return 1
  jr_jobs_dir_for "$arm" || return 2
  for t in "$@"; do only+=(--only "$t"); done
  [ "$arm" != "all" ] && armargs=(--arms "$arm")

  local S="$PKG_ROOT/eval/benchmir"
  case "$GRID" in
    pass2)
      # Point the grid at the snapshots run_snapshot.sh froze, resolving the
      # <arm>.ckpt symlink to its stamped target so the generated job configs
      # record the STEP they evaluated, not a name that moves.
      local ck=() a
      for a in $JR_ARMS; do
        jr_ckpt_arg "$(jr_eval_arm "$a")" "$a" && ck+=("${JR_CKPT_ARG[@]}")
      done
      "$py" "$S/scripts_ours/gen_jobs_pass2.py" \
          --jobs-dir "$JR_JOBS_DIR" --out-runs "$RUN_ROOT/pass2" \
          "${armargs[@]}" "${only[@]}" "${ck[@]}" ;;
    composer)
      "$py" "$S/scripts_ours/gen_jobs_composer.py" \
          --jobs-dir "$JR_JOBS_DIR" --out-runs "$RUN_ROOT/composer" \
          "${armargs[@]}" ;;
    v1|cipi_official|pop909full|topmagd)
      # These grids have a fixed arm list and a fixed task list by construction:
      # v1 IS the reference column, and the other three exist to answer one
      # question each. Say so rather than accepting a restriction and ignoring it.
      [ "$arm" != "all" ] && echo "note: GRID=$GRID generates every arm; '$arm' ignored" >&2
      [ $# -gt 0 ] && echo "note: GRID=$GRID has a fixed task list; $* ignored" >&2
      case "$GRID" in
        v1)            "$py" "$S/scripts_ours/gen_jobs.py" ;;
        cipi_official) "$py" "$S/scripts_ours/gen_jobs_cipi_official.py" ;;
        pop909full)    "$py" "$S/scripts_ours/gen_jobs_pop909full.py" ;;
        topmagd)
          # Lifts each arm's model entry out of the pass2/v1 job configs, so
          # those grids must have been generated first.
          "$py" "$S/scripts_topmagd/gen_jobs_topmagd_splits.py" \
              --jobs-dir "$JR_JOBS_DIR" --out-runs "$RUN_ROOT/topmagd_splits" ;;
      esac ;;
  esac
}

jr_ckpt_arg() {                  # $1 = eval arm name, $2 = training arm name
  local link="$CHECKPOINT_ROOT/$2.ckpt" real
  JR_CKPT_ARG=()
  [ -e "$link" ] || return 1
  real="$(readlink -f "$link")"
  JR_CKPT_ARG=(--ckpt "$1=$real")
}
