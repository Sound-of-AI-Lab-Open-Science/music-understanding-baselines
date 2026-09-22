#!/bin/bash
# One-time setup: the submodule, the three conda environments, the evaluation
# library, and (optionally) the upstream MuseTok checkout.
#
#   ./setup.sh                 # everything
#   ./setup.sh --envs-only     # skip the submodule and MuseTok
#   ./setup.sh --with-musetok  # also clone the upstream MuseTok checkout
#
# Re-runnable: an environment that already exists is left alone, and the
# editable install is idempotent.
#
# WHY THREE ENVIRONMENTS. Their torch builds are mutually exclusive, not merely
# different: the MuseTok checkpoints need miditoolkit 1.0.0 (1.0.1 changes tick
# handling and silently produces a DIFFERENT REMI+ encoding), and the evaluation
# library pins a torch with no CUDA wheel for many drivers. So the adapters run
# each model in a SUBPROCESS under its own interpreter and talk to it over a
# framed protocol. That is also why the probes are CPU jobs and why embedding
# extraction is pushed out into the two GPU-capable environments.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./env.sh

DO_SUBMODULE=1; DO_MUSETOK=0
for a in "$@"; do
  case "$a" in
    --envs-only)    DO_SUBMODULE=0 ;;
    --with-musetok) DO_MUSETOK=1 ;;
    *) echo "usage: ./setup.sh [--envs-only] [--with-musetok]" >&2; exit 2 ;;
  esac
done

if [ "$DO_SUBMODULE" = "1" ]; then
  echo "== evaluation library submodule"
  # Not fatal. Whether the evaluation library's fork is published is the
  # BenchMIR authors' decision (see THIRD_PARTY_NOTICES.md), so the URL may not
  # resolve for you. The whole pre-training half runs without it; only
  # ./run_eval.sh and ./run_report.sh need it.
  #
  # Prompting is disabled for this one fetch, over both transports: without it
  # an unreadable submodule asks for credentials on the terminal and hangs a
  # non-interactive setup instead of falling through to the warning below.
  if GIT_TERMINAL_PROMPT=0 \
     GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh} -o BatchMode=yes" \
     git -C "$PKG_ROOT" submodule update --init --recursive -- "$BENCHMIR_ROOT" 2>/dev/null \
     || GIT_TERMINAL_PROMPT=0 \
        GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh} -o BatchMode=yes" \
        git -C "$PKG_ROOT" submodule update --init --recursive 2>/dev/null; then
    echo "  ok"
  else
    echo "  WARNING: the evaluation library submodule could not be fetched."
    echo "  This is expected if you have no access to the fork, and is not an error:"
    echo "  pre-training still works, evaluation does not, and run_smoke.sh will say"
    echo "  'skipped' rather than 'FAIL'. See THIRD_PARTY_NOTICES.md."
  fi
fi

CONDA="$CONDA_ROOT/bin/conda"
[ -x "$CONDA" ] || CONDA="$(command -v conda || true)"
[ -n "$CONDA" ] || { echo "conda not found; set CONDA_ROOT" >&2; exit 1; }

echo "== conda environments"
create_env () {                 # $1 = env name, $2 = spec file
  if [ -d "$CONDA_ROOT/envs/$1" ]; then
    echo "  $1 exists, leaving it alone"
  else
    echo "  creating $1 from $2"
    "$CONDA" env create -n "$1" -f "$PKG_ROOT/envs/$2"
  fi
}
create_env "$JEPA_ENV"     jepa-music.yml
create_env "$MUSETOK_ENV"  musetok.yml
create_env "$BENCHMIR_ENV" benchmir.yml

echo "== evaluation library (editable install into $BENCHMIR_ENV)"
if [ -f "$BENCHMIR_ROOT/pyproject.toml" ]; then
  "$CONDA_ROOT/envs/$BENCHMIR_ENV/bin/pip" install -e "$BENCHMIR_ROOT"
else
  echo "  SKIPPED: $BENCHMIR_ROOT is empty (submodule not fetched)."
  echo "  Evaluation is unavailable; pre-training is unaffected."
fi

if [ "$DO_MUSETOK" = "1" ]; then
  # Upstream MuseTok publishes NO licence file, so this package neither vendors
  # it nor registers it as a submodule; it is cloned here, at a pinned commit,
  # only if you ask for it. See THIRD_PARTY_NOTICES.md. Without it, the MuseTok
  # arm cannot run; the other three arms are unaffected.
  MUSETOK_URL="${MUSETOK_URL:-https://github.com/Yuer867/MuseTok.git}"
  MUSETOK_PIN="${MUSETOK_PIN:-7b71d16868431331386b1c1088193b3b85bd70e2}"
  echo "== MuseTok upstream checkout -> $MUSETOK_REPO"
  if [ -d "$MUSETOK_REPO/.git" ]; then
    echo "  exists, leaving it alone"
  else
    git clone "$MUSETOK_URL" "$MUSETOK_REPO"
    git -C "$MUSETOK_REPO" checkout "$MUSETOK_PIN"
  fi
else
  echo "== MuseTok upstream checkout: skipped (pass --with-musetok)"
fi

echo
echo "done. next: ./check_data.sh, then ./run_smoke.sh"
