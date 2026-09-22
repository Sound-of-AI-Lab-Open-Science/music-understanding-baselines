# Shell configuration for the whole package.  Source it, never execute it:
#
#     source /path/to/this/package/env.sh
#
# Every entry point (run_*.sh) and every SLURM template sources this file, so
# there is exactly one place where a path is decided.  Defaults come from
# paths.yaml; an environment variable that is already set always wins, so a
# site adapts the package by exporting variables, not by editing files.
#
#     export WORK_ROOT=/scratch/$USER/jepa-reproduce
#     export DATA_ROOT=/datasets/symbolic-music
#     source env.sh && ./check_data.sh
#
# shellcheck shell=bash

# ---------------------------------------------------------------------------
# PKG_ROOT: the directory holding this file, resolved through symlinks so the
# package works when it is linked into a scratch tree.
# ---------------------------------------------------------------------------
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _JR_SELF="${BASH_SOURCE[0]}"
else
  _JR_SELF="$0"
fi
PKG_ROOT="$(cd "$(dirname "$(readlink -f "$_JR_SELF")")" && pwd)"
export PKG_ROOT
unset _JR_SELF

# ---------------------------------------------------------------------------
# paths.yaml reader.  The file is deliberately flat (`key: value`), so a two
# line parser is enough and the package needs no YAML tool in the shell.
# ---------------------------------------------------------------------------
_jr_yaml_get() {                 # $1 = key -> raw value, quotes and comment stripped
  sed -n "s/^$1:[[:space:]]*//p" "$PKG_ROOT/paths.yaml" | head -1 \
    | sed -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//'
}

_jr_default() {                  # $1 = VAR, $2 = paths.yaml key
  local raw
  if [ -z "${!1:-}" ]; then
    raw="$(_jr_yaml_get "$2")"
    eval "$1=\"$raw\""           # expands ${PKG_ROOT}, ${HOME}, earlier keys
  fi
  export "${1?}"
}

# Resolution order matters: a key may reference any key resolved above it.
_jr_default DATA_ROOT           data_root
_jr_default BENCHMIR_DATA_ROOT  benchmir_data_root
_jr_default WORK_ROOT           work_root
_jr_default CHECKPOINT_ROOT     checkpoint_root
_jr_default CONDA_ROOT          conda_root
# Empty in paths.yaml by default: derive it from the conda installation that is
# actually running, so nothing has to be edited on a new machine. A batch job
# often has no conda on PATH, so conventional install prefixes are probed too.
# Set $CONDA_ROOT (or conda_root in paths.yaml) if none of this finds yours.
if [ -z "${CONDA_ROOT:-}" ]; then
  if [ -n "${CONDA_EXE:-}" ]; then
    CONDA_ROOT="$(dirname "$(dirname "$CONDA_EXE")")"
  elif [ -n "${CONDA_PREFIX:-}" ]; then
    case "$CONDA_PREFIX" in
      */envs/*) CONDA_ROOT="${CONDA_PREFIX%/envs/*}" ;;
      *)        CONDA_ROOT="$CONDA_PREFIX" ;;
    esac
  else
    _jr_conda="$(command -v conda || true)"
    if [ -n "$_jr_conda" ]; then
      CONDA_ROOT="$(dirname "$(dirname "$_jr_conda")")"
    else
      for _jr_c in "$HOME"/mini*3 "$HOME"/ana*3 /opt/conda /usr/local/conda; do
        [ -x "$_jr_c/bin/conda" ] && { CONDA_ROOT="$_jr_c"; break; }
      done
      unset _jr_c
    fi
    unset _jr_conda
  fi
  export CONDA_ROOT
fi

_jr_default JEPA_ENV            jepa_env
_jr_default MUSETOK_ENV         musetok_env
_jr_default MIDI_RAE_ENV        midi_rae_env
_jr_default BENCHMIR_ENV        benchmir_env
_jr_default BENCHMIR_ROOT       benchmir_root
_jr_default MUSETOK_REPO        musetok_repo
_jr_default JR_SLURM_PARTITION     slurm_partition
_jr_default JR_SLURM_GPU_PARTITION slurm_gpu_partition
_jr_default JR_SLURM_ACCOUNT       slurm_account
_jr_default JR_SLURM_QOS           slurm_qos
_jr_default JR_SLURM_EXCLUDE       slurm_exclude
_jr_default JR_SLURM_GRES          slurm_gres

# ---------------------------------------------------------------------------
# Derived locations.  These are the names the code actually reads.
# ---------------------------------------------------------------------------
: "${CACHE_ROOT:=${WORK_ROOT}/cache}"          # per-shard and union token caches
: "${SPLIT_ROOT:=${WORK_ROOT}/splits}"         # 98/1/1 content-hash song splits
# One corpus variant = one split tag = one union cache suffix.  Changing the
# tag is how a second corpus (a keep-list applied, say) gets its own split
# files and its own union caches without touching the first one's -- the merge
# WRITES pieces_{train,val}.txt into the split dir it is given, so two variants
# sharing a split dir would silently overwrite each other's piece lists.
: "${SPLIT_TAG:=main}"
: "${SPLIT_DIR:=${SPLIT_ROOT}/${SPLIT_TAG}}"
: "${VOCAB_ROOT:=${WORK_ROOT}/vocab}"          # corpus-derived MuseTok dictionary
: "${RUN_ROOT:=${WORK_ROOT}/runs}"             # training runs and evaluation runs
: "${JOBS_ROOT:=${WORK_ROOT}/jobs}"            # generated per-job eval configs
: "${LOG_ROOT:=${WORK_ROOT}/logs}"             # job logs
: "${MIDI_DIR:=${DATA_ROOT}/musescore}"        # pre-training corpus (recursive)
export CACHE_ROOT SPLIT_ROOT SPLIT_TAG SPLIT_DIR VOCAB_ROOT RUN_ROOT
export JOBS_ROOT LOG_ROOT MIDI_DIR

# ---------------------------------------------------------------------------
# The contract the BenchMIR model adapters read (third_party/BenchMIR).  Each
# has a relative fallback inside the library; setting them here is what makes
# an out-of-tree working directory safe.
# ---------------------------------------------------------------------------
: "${CONDA_ENVS_ROOT:=${CONDA_ROOT}/envs}"     # where the worker interpreters live
: "${BENCHMIR_OURS_CACHE:=${WORK_ROOT}/cache/embeddings}"
: "${JEPA_ROOT:=${PKG_ROOT}/pretrain}"         # holds the src/ package the workers import
: "${MUSICBERT_ENV:=${JEPA_ENV}}"
: "${JEPA_CKPT:=${CHECKPOINT_ROOT}/music_jepa_final.pt}"
# A SECOND JEPA pre-training seed, supplied by you exactly like $JEPA_CKPT.  The
# `v1` grid uses it as a diagnostic control: it is the run that shows a collapsed
# encoder is a property of $JEPA_CKPT rather than of the adapter code.  Nothing
# in this package produces it, and only the `v1` grid reads it.
: "${JEPA_CKPT_SEED2:=${CHECKPOINT_ROOT}/music_jepa_champA_s12.pt}"
: "${MUSICBERT_CKPT:=${CHECKPOINT_ROOT}/musicbert_base_converted.pt}"
: "${MUSETOK_CKPT:=${MUSETOK_REPO}/ckpt/best_tokenizer/model.pt}"
: "${MUSETOK_MUSESCORE_VOCAB:=${VOCAB_ROOT}/dictionary_musescore.pkl}"
export CONDA_ENVS_ROOT BENCHMIR_OURS_CACHE JEPA_ROOT MUSICBERT_ENV
export JEPA_CKPT JEPA_CKPT_SEED2 MUSICBERT_CKPT MUSETOK_CKPT MUSETOK_MUSESCORE_VOCAB

# `baselines/train.py` is launched as a script, so sys.path[0] is baselines/;
# pretrain/ must also be importable for `import src.*` and `import baselines.*`.
case ":${PYTHONPATH:-}:" in
  *":${PKG_ROOT}/pretrain:"*) : ;;
  *) PYTHONPATH="${PKG_ROOT}/pretrain${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
export PYTHONPATH
export PYTHONIOENCODING=utf-8

# ---------------------------------------------------------------------------
# Helpers used by the entry points and the SLURM templates.
# ---------------------------------------------------------------------------

#: The pre-training arms.  An "arm" is one (model, recipe, corpus codec)
#: triple; everything downstream -- the config, the interpreter, the union
#: cache, the run directory, the checkpoint name -- is derived from its name, so
#: nothing has to be kept in sync by hand.
#: midi_rae_enc/midi_rae_dec added alongside the original four -- additive
#: only, the four above are untouched.
JR_ARMS="jepa_paper jepa_champA musetok musicbert midi_rae_enc midi_rae_dec"
export JR_ARMS

jr_arm_model() {                 # arm -> the model train.py implements
  case "$1" in
    jepa_paper|jepa_champA) printf 'jepa' ;;
    musetok)                printf 'musetok' ;;
    musicbert)              printf 'musicbert' ;;
    midi_rae_enc)            printf 'midi_rae_enc' ;;
    midi_rae_dec)            printf 'midi_rae_dec' ;;
    *) echo "unknown arm '$1' (have: $JR_ARMS)" >&2; return 2 ;;
  esac
}

jr_arm_codec() {                 # arm -> which tokenization it consumes
  case "$1" in
    musetok)                          printf 'remi' ;;
    jepa_paper|jepa_champA|musicbert) printf 'octuple' ;;
    midi_rae_enc|midi_rae_dec)        printf 'pianoroll' ;;
    *) echo "unknown arm '$1' (have: $JR_ARMS)" >&2; return 2 ;;
  esac
}

jr_arm_env() {                   # arm -> conda environment name
  case "$1" in
    musetok)                          printf '%s' "$MUSETOK_ENV" ;;
    jepa_paper|jepa_champA)           printf '%s' "$JEPA_ENV" ;;
    musicbert)                        printf '%s' "$MUSICBERT_ENV" ;;
    midi_rae_enc|midi_rae_dec)        printf '%s' "$MIDI_RAE_ENV" ;;
    *) echo "unknown arm '$1' (have: $JR_ARMS)" >&2; return 2 ;;
  esac
}

jr_union_cache() {               # codec -> the union cache this corpus variant uses
  printf '%s/union_%s_%s' "$CACHE_ROOT" "$1" "$SPLIT_TAG"
}

# jr_python <env-name> -> absolute interpreter path for that conda environment.
jr_python() {
  local env="$1" py="${CONDA_ROOT}/envs/$1/bin/python"
  if [ -z "${CONDA_ROOT:-}" ]; then
    echo "CONDA_ROOT is empty: no conda installation could be located." >&2
    echo "Export CONDA_ROOT, or set conda_root in paths.yaml." >&2
    return 1
  fi
  [ -x "$py" ] || { echo "no interpreter for conda env '$env' at $py" >&2; return 1; }
  printf '%s' "$py"
}

# jr_sbatch_args [--gpu] -> the cluster flags every submission shares, as words
# on stdout.  An empty knob contributes no flag at all, so the package runs on a
# cluster that has no partition/QOS/account policy without edits.
jr_sbatch_args() {
  local part="${JR_SLURM_PARTITION}"
  # --export=ALL is NOT redundant. Slurm's own default is ALL, but a site can
  # turn that off for every user at once by exporting SBATCH_EXPORT=NONE from
  # the login profile -- and several do. Every template in slurm/ begins by
  # reading $PKG_ROOT out of the environment and sourcing env.sh from it, and
  # every override a run is configured with -- $WORK_ROOT, $MIDI_DIR,
  # $BENCHMIR_DATA_ROOT, $SPLIT_TAG -- reaches the job the same way. So a
  # stripped environment is not a degraded run: the job dies on its first line
  # with "PKG_ROOT: source env.sh before submitting". A flag on the command line
  # beats SBATCH_EXPORT, so stating it here is what makes the package behave the
  # same on both kinds of cluster.
  printf -- '--export=ALL '
  if [ "${1:-}" = "--gpu" ]; then
    part="${JR_SLURM_GPU_PARTITION:-$JR_SLURM_PARTITION}"
    [ -n "${JR_SLURM_GRES}" ] && printf -- '--gres=%s ' "${JR_SLURM_GRES}"
  fi
  [ -n "$part" ]             && printf -- '--partition=%s ' "$part"
  [ -n "${JR_SLURM_ACCOUNT}" ]  && printf -- '--account=%s '   "${JR_SLURM_ACCOUNT}"
  [ -n "${JR_SLURM_QOS}" ]      && printf -- '--qos=%s '       "${JR_SLURM_QOS}"
  [ -n "${JR_SLURM_EXCLUDE}" ]  && printf -- '--exclude=%s '   "${JR_SLURM_EXCLUDE}"
  return 0
}

# jr_require <path> [message] -> fail loudly, early, with the variable to set.
jr_require() {
  [ -e "$1" ] && return 0
  echo "missing: $1${2:+  ($2)}" >&2
  return 1
}

# jr_prepare_dirs -> create the writable tree.  Sourcing env.sh has no side
# effects on the filesystem; the entry points call this once, up front.
jr_prepare_dirs() {
  mkdir -p "${WORK_ROOT}" "${CACHE_ROOT}" "${SPLIT_ROOT}" "${VOCAB_ROOT}" \
           "${RUN_ROOT}" "${JOBS_ROOT}" "${LOG_ROOT}" "${CHECKPOINT_ROOT}" \
           "${BENCHMIR_OURS_CACHE}"
}
