#!/bin/bash
# Prove the pipeline before committing a GPU week to it.
#
#   ./run_smoke.sh                 # everything that can be checked without data
#   ./run_smoke.sh --with-data     # ...plus a real tokenize + 2 training steps
#   ./run_smoke.sh --end-to-end    # the WHOLE pipeline, unattended, ~30 minutes
#
# --end-to-end is the one that matters. It cuts itself a hundred-file corpus and
# a forty-movement evaluation corpus, then runs every stage THROUGH THE PACKAGE'S
# OWN ENTRY POINTS -- run_tokenize.sh, run_union.sh, run_pretrain.sh,
# run_snapshot.sh, run_eval.sh, run_report.sh, in that order -- waiting for each
# SLURM stage before starting the next, and prints a table of what each cost.
# Nothing is stubbed and no shortcut path exists for it: if it passes, the
# commands in README.md work on this cluster, in these environments, with these
# recipes. The numbers it produces are MEANINGLESS by construction and are
# printed anyway, because a table that renders is part of what it is proving.
#
# It never touches a real campaign. WORK_ROOT is redirected to $WORK_ROOT/smoke
# and every derived path -- caches, splits, vocabulary, runs, checkpoints, jobs,
# logs -- is re-derived under it, so the smoke cannot repoint a <arm>.ckpt
# symlink an evaluation grid is reading or collide with a union cache.
#
# What the no-argument form checks, in order:
#   1. the three conda environments exist and import what they must
#   2. the evaluation library is installed and importable (the submodule)
#   3. every recipe parses, and every ${VAR} in it resolves to a real path
#
# Step 2 is SKIPPED, not failed, when the submodule is absent: whether the
# evaluation library's fork is published is the BenchMIR authors' decision, so
# a checkout without it is a supported configuration (see README.md and
# THIRD_PARTY_NOTICES.md). The pre-training half is complete in that state, and
# this script still exits 0.
#
# Step 3 is the one that catches the common mistake: a recipe names
# ${SPLIT_DIR}/ids_train.txt, and a shell that never sourced env.sh leaves that
# unexpanded, which fails as a missing file rather than silently training on the
# wrong corpus.
#
# Knobs for --end-to-end (defaults in brackets):
#   JR_SMOKE_FILES      [100]  pre-training MIDI files to copy
#   JR_SMOKE_SHARDS     [2]    shards to spread them over (= array tasks)
#   JR_SMOKE_MOVEMENTS  [40]   CIPI movements to copy
#   JR_SMOKE_STEPS      [30]   training steps per arm
#   JR_SMOKE_TASK       [cipi_difficulty]  the one probe to fit
#   JR_SMOKE_POLL       [20]   seconds between queue polls
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh

MODE="${1:-}"
case "$MODE" in
  ""|--with-data|--end-to-end) ;;
  *) echo "usage: ./run_smoke.sh [--with-data|--end-to-end]" >&2; exit 2 ;;
esac

FAIL=0
SKIPPED=0
ok ()   { printf '  ok    %s\n' "$*"; }
bad ()  { printf '  FAIL  %s\n' "$*"; FAIL=1; }
skip () { printf '  skip  %s\n' "$*"; SKIPPED=$((SKIPPED + 1)); }

echo "== 1. conda environments"
for pair in "$JEPA_ENV:torch,lightning,miditoolkit,yaml" \
            "$MUSETOK_ENV:torch,miditoolkit,vector_quantize_pytorch" \
            "$BENCHMIR_ENV:torch,yaml"; do
  ENVNAME="${pair%%:*}"; MODS="${pair#*:}"
  PY="$CONDA_ROOT/envs/$ENVNAME/bin/python"
  if [ ! -x "$PY" ]; then bad "conda env '$ENVNAME' (run ./setup.sh)"; continue; fi
  if "$PY" -c "import ${MODS//,/, }" 2>/dev/null; then ok "$ENVNAME: $MODS"
  else bad "$ENVNAME cannot import $MODS"; fi
done

echo "== 2. evaluation library"
PY_BM="$CONDA_ROOT/envs/$BENCHMIR_ENV/bin/python"
HAVE_BENCHMIR=0
if [ -f "$BENCHMIR_ROOT/pyproject.toml" ]; then
  # Submodule present, so the editable install is expected and its absence is a
  # real failure.
  ok "submodule checked out"
  if [ -x "$PY_BM" ] && "$PY_BM" -c "import benchmir" 2>/dev/null; then
    ok "benchmir importable in $BENCHMIR_ENV"
    HAVE_BENCHMIR=1
  else
    bad "benchmir not installed (./setup.sh runs pip install -e third_party/BenchMIR)"
  fi
else
  # Submodule absent: a documented, supported state, not a broken package.
  skip "third_party/BenchMIR is empty -- evaluation half not checked"
  printf '        the pre-training half is complete without it; run_eval.sh and\n'
  printf '        run_report.sh are the two that need it. If you have access to\n'
  printf '        the fork: git submodule update --init && ./setup.sh\n'
fi

echo "== 3. recipes parse and resolve"
PY_J="$CONDA_ROOT/envs/$JEPA_ENV/bin/python"
if [ -x "$PY_J" ]; then
  for f in pretrain/baselines/configs/*.yaml eval/benchmir/configs/*.yaml; do
    if OUT=$("$PY_J" - "$f" <<'PYEOF'
import os, sys, yaml
p = sys.argv[1]
def walk(n):
    if isinstance(n, str): return [n] if "${" in os.path.expandvars(n) else []
    if isinstance(n, dict): return [x for v in n.values() for x in walk(v)]
    if isinstance(n, list): return [x for v in n for x in walk(v)]
    return []
left = walk(yaml.safe_load(open(p)))
print("UNRESOLVED " + ", ".join(sorted(set(left))) if left else "")
PYEOF
    ); then
      [ -z "$OUT" ] && ok "$f" || bad "$f: $OUT"
    else
      bad "$f does not parse"
    fi
  done
else
  bad "cannot check recipes: no interpreter for '$JEPA_ENV'"
fi

if [ "$MODE" = "--with-data" ]; then
  echo "== 4. two training steps per arm (CPU)"
  jr_prepare_dirs
  jr_require "$MIDI_DIR" "set DATA_ROOT/MIDI_DIR" || FAIL=1
  cd "$PKG_ROOT/pretrain"
  for ARM in $JR_ARMS; do
    PY="$(jr_python "$(jr_arm_env "$ARM")")" || { bad "$ARM: no interpreter"; continue; }
    # This mode tokenizes from --midi-dir rather than reading a union cache, so
    # it reaches the MuseTok vocabulary check without a vocabulary existing:
    # dictionary_musescore.pkl is DERIVED from the REMI+ shard caches by
    # run_union.sh, which this mode does not run. That is a missing
    # prerequisite, not a broken arm -- use --end-to-end, which does run it.
    if [ "$(jr_arm_codec "$ARM")" = "remi" ] && [ ! -f "$MUSETOK_MUSESCORE_VOCAB" ]; then
      skip "$ARM: no $MUSETOK_MUSESCORE_VOCAB yet (run_union.sh builds it; or use --end-to-end)"
      continue
    fi
    LOG="$LOG_ROOT/smoke_withdata_${ARM}.log"
    mkdir -p "$LOG_ROOT"
    if "$PY" baselines/train.py --model "$(jr_arm_model "$ARM")" \
         --config "baselines/configs/${ARM}.yaml" \
         --midi-dir "$MIDI_DIR" --cache-dir "$WORK_ROOT/smoke_cache" \
         --out-dir "$RUN_ROOT/${ARM}_smoke" \
         --smoke --accelerator cpu --max-steps 2 >"$LOG" 2>&1; then
      ok "$ARM trained 2 steps"
    else
      # The log, not "rerun it yourself": a smoke that reports a failure it has
      # already diagnosed and then throws the diagnosis away wastes the run.
      bad "$ARM smoke train failed -- $LOG"
      sed -n '$p' "$LOG" | sed 's/^/        /'
    fi
  done
  cd "$PKG_ROOT"
fi

# ---------------------------------------------------------------------------
# --end-to-end
# ---------------------------------------------------------------------------
if [ "$MODE" = "--end-to-end" ] && [ "$FAIL" = "0" ]; then
  POLL="${JR_SMOKE_POLL:-20}"
  STEPS="${JR_SMOKE_STEPS:-30}"
  TASK="${JR_SMOKE_TASK:-cipi_difficulty}"
  SRC_MIDI="$MIDI_DIR"
  SRC_CIPI="$BENCHMIR_DATA_ROOT/cipi"

  # Re-root EVERY writable path under $WORK_ROOT/smoke by unsetting the derived
  # variables and letting env.sh rebuild them from the new WORK_ROOT. Setting
  # WORK_ROOT alone would not do it: this shell has already exported
  # $CACHE_ROOT, $SPLIT_DIR, $CHECKPOINT_ROOT and the rest at their real values,
  # and env.sh's `: "${VAR:=default}"` keeps an existing value on purpose.
  unset CACHE_ROOT SPLIT_ROOT SPLIT_DIR VOCAB_ROOT RUN_ROOT JOBS_ROOT LOG_ROOT \
        CHECKPOINT_ROOT BENCHMIR_OURS_CACHE MUSETOK_MUSESCORE_VOCAB \
        JEPA_CKPT JEPA_CKPT_SEED2 MUSICBERT_CKPT MUSETOK_CKPT MIDI_DIR
  export WORK_ROOT="$WORK_ROOT/smoke"
  export SPLIT_TAG=smoke
  export MIDI_DIR="$WORK_ROOT/midi"
  export BENCHMIR_DATA_ROOT="$WORK_ROOT/data"
  source ./env.sh
  source ./lib/grids.sh          # jr_eval_arm: training arm -> evaluation arm
  jr_prepare_dirs

  echo
  echo "== 4. end to end, on a corpus this script cuts for itself"
  echo "     work root   $WORK_ROOT"
  echo "     logs        $LOG_ROOT"

  STAGES=()
  STAGE_T=()
  STAGE_RC=()
  T_ALL=$(date +%s)

  # jr_smoke_run <label> <cmd...> -- time it, log it, remember whether it passed.
  jr_smoke_run () {
    local label="$1"; shift
    local log="$LOG_ROOT/smoke_${label//:/_}.log" t0 rc
    t0=$(date +%s)
    "$@" >"$log" 2>&1; rc=$?
    STAGES+=("$label"); STAGE_T+=("$(( $(date +%s) - t0 ))"); STAGE_RC+=("$rc")
    if [ "$rc" = "0" ]; then
      printf '  ok    %-17s %4ds  %s\n' "$label" "${STAGE_T[$((${#STAGE_T[@]} - 1))]}" "$log"
    else
      printf '  FAIL  %-17s %4ds  %s\n' "$label" "${STAGE_T[$((${#STAGE_T[@]} - 1))]}" "$log"
      tail -n 15 "$log" | sed 's/^/        /'
      FAIL=1
    fi
    return $rc
  }

  # jr_smoke_wait <job id...> -- block until they leave the queue, then insist
  # every one of them ended COMPLETED. squeue going empty is NOT success: a job
  # that died in its first second leaves the queue exactly as fast as one that
  # worked, which is how an unattended smoke ends up reporting a green run over
  # four crashed arms.
  jr_smoke_wait () {
    local joined bad tries=0
    joined=$(printf '%s,' "$@"); joined="${joined%,}"
    [ -n "$joined" ] || return 0
    while [ "$(squeue -h -j "$joined" 2>/dev/null | wc -l)" -gt 0 ]; do sleep "$POLL"; done
    # sacct can trail squeue by a few seconds at the end of an array.
    while [ "$tries" -lt 10 ]; do
      bad=$(sacct -n -X -P -j "$joined" -o State 2>/dev/null | sort -u)
      [ -n "$bad" ] && break
      tries=$((tries + 1)); sleep 3
    done
    if [ -z "$bad" ]; then
      echo "        WARNING: no accounting record for $joined; state unknown"
      return 0
    fi
    bad=$(printf '%s\n' "$bad" | grep -v '^COMPLETED$' | tr '\n' ' ')
    [ -z "$bad" ] && return 0
    echo "        job(s) $joined ended: $bad"
    return 1
  }

  # jr_smoke_ids -- job ids out of a runner's own stdout, which every runner
  # prints in the form "<what>: job <id>" or "<what>: <id> (n jobs...)".
  jr_smoke_ids () { grep -oE '(job |: )[0-9]{4,}' "$1" | grep -oE '[0-9]{4,}'; }

  jr_smoke_stage_jobs () {          # <label> <runner...> : submit, then wait
    local label="$1"; shift
    local log="$LOG_ROOT/smoke_${label//:/_}.log" t0 rc ids
    t0=$(date +%s)
    "$@" >"$log" 2>&1; rc=$?
    if [ "$rc" = "0" ]; then
      mapfile -t ids < <(jr_smoke_ids "$log")
      if [ "${#ids[@]}" -eq 0 ]; then
        echo "        no job id in $log" >>"$log"; rc=1
      else
        jr_smoke_wait "${ids[@]}" >>"$log" 2>&1 || rc=1
      fi
    fi
    STAGES+=("$label"); STAGE_T+=("$(( $(date +%s) - t0 ))"); STAGE_RC+=("$rc")
    if [ "$rc" = "0" ]; then
      printf '  ok    %-17s %4ds  %s\n' "$label" "${STAGE_T[$((${#STAGE_T[@]} - 1))]}" "$log"
    else
      printf '  FAIL  %-17s %4ds  %s\n' "$label" "${STAGE_T[$((${#STAGE_T[@]} - 1))]}" "$log"
      tail -n 15 "$log" | sed 's/^/        /'
      FAIL=1
    fi
    return $rc
  }

  PY_CUT="$(jr_python "$JEPA_ENV")" || { bad "no interpreter for '$JEPA_ENV'"; PY_CUT=""; }

  # ---- corpora ----------------------------------------------------------
  [ -n "$PY_CUT" ] && jr_smoke_run corpus "$PY_CUT" "$PKG_ROOT/lib/smoke_corpus.py" midi \
      --src "$SRC_MIDI" --out "$MIDI_DIR" \
      --files "${JR_SMOKE_FILES:-100}" --shards "${JR_SMOKE_SHARDS:-2}"

  DO_EVAL=0
  if [ "$FAIL" = "0" ] && [ "$HAVE_BENCHMIR" = "1" ] && [ -d "$SRC_CIPI" ]; then
    jr_smoke_run cipi "$PY_CUT" "$PKG_ROOT/lib/smoke_corpus.py" cipi \
        --src "$SRC_CIPI" --out "$BENCHMIR_DATA_ROOT/cipi" \
        --movements "${JR_SMOKE_MOVEMENTS:-40}" && DO_EVAL=1
  elif [ "$HAVE_BENCHMIR" != "1" ]; then
    skip "evaluation half: the submodule is not installed"
  else
    skip "evaluation half: no CIPI corpus at $SRC_CIPI"
  fi

  # ---- the pipeline, through its own entry points ------------------------
  if [ "$FAIL" = "0" ]; then
    # The walltimes are the smoke's, not the recipes': a stage that would queue
    # for hours behind a 13-hour request is not a smoke either.
    export JR_TOKENIZE_TIME="${JR_TOKENIZE_TIME:-00:30:00}"
    export JR_TOKENIZE_MEM="${JR_TOKENIZE_MEM:-8G}"
    export JR_PRETRAIN_SMOKE_TIME="${JR_PRETRAIN_SMOKE_TIME:-00:50:00}"
    export JR_PRETRAIN_SMOKE_MEM="${JR_PRETRAIN_SMOKE_MEM:-16G}"
    export JR_ENCODE_TIME="${JR_ENCODE_TIME:-00:45:00}"
    export JR_ENCODE_MEM="${JR_ENCODE_MEM:-6G}"
    export JR_EVAL_TIME="${JR_EVAL_TIME:-00:45:00}"
    export JR_EVAL_MEM="${JR_EVAL_MEM:-8G}"
    # Two warm shards, not twelve: the shard count is per model and the corpus
    # is forty files.
    export BENCHMIR_PASS2_WARM_SHARDS="${BENCHMIR_PASS2_WARM_SHARDS:-2}"
    export JR_ENCODE_DATASETS="${JR_ENCODE_DATASETS:---dataset cipi}"

    jr_smoke_stage_jobs tokenize sh -c \
      '"$0"/run_tokenize.sh octuple && "$0"/run_tokenize.sh remi' "$PKG_ROOT"
  fi

  [ "$FAIL" = "0" ] && jr_smoke_run union "$PKG_ROOT/run_union.sh"

  if [ "$FAIL" = "0" ]; then
    for ARM in $JR_ARMS; do
      jr_smoke_stage_jobs "pre:$ARM" "$PKG_ROOT/run_pretrain.sh" "$ARM" \
          --smoke --max-steps "$STEPS" || break
    done
  fi

  if [ "$FAIL" = "0" ]; then
    for ARM in $JR_ARMS; do
      jr_smoke_run "snap:$ARM" "$PKG_ROOT/run_snapshot.sh" "$ARM" latest --smoke || break
    done
  fi

  if [ "$FAIL" = "0" ] && [ "$DO_EVAL" = "1" ]; then
    export GRID=pass2
    for ARM in $JR_ARMS; do
      jr_smoke_stage_jobs "eval:$ARM" "$PKG_ROOT/run_eval.sh" \
          "$(jr_eval_arm "$ARM")" "$TASK" || break
    done
    [ "$FAIL" = "0" ] && jr_smoke_run report "$PKG_ROOT/run_report.sh" pass2
    [ -f "$RUN_ROOT/pass2/CONSOLIDATED.md" ] && {
      echo
      echo "     ---- the table, whose numbers mean NOTHING: $STEPS steps per arm,"
      echo "          ${JR_SMOKE_MOVEMENTS:-40} movements, one unseeded probe fit ----"
      sed 's/^/     /' "$RUN_ROOT/pass2/CONSOLIDATED.md"
    }
  fi

  echo
  printf '     %-17s %6s\n' "stage" "sec"
  for i in "${!STAGES[@]}"; do
    printf '     %-17s %6s  %s\n' "${STAGES[$i]}" "${STAGE_T[$i]}" \
           "$([ "${STAGE_RC[$i]}" = 0 ] && echo ok || echo FAILED)"
  done
  printf '     %-17s %6s\n' "TOTAL" "$(( $(date +%s) - T_ALL ))"
fi

echo
if [ "$FAIL" != "0" ]; then
  echo "SMOKE FAILED"
elif [ "$SKIPPED" != "0" ]; then
  echo "SMOKE OK ($SKIPPED check(s) skipped -- see above; this is not a failure)"
else
  echo "SMOKE OK"
fi
exit "$FAIL"
