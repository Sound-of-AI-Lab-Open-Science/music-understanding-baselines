#!/bin/bash
#SBATCH --job-name=jr_snapshot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
# ^ last-resort default only. Slurm resolves it against the SUBMISSION
# directory, so it works when submitted from $WORK_ROOT (what the run_*.sh
# runners do) and nowhere else. Every runner overrides it with an absolute
# --output under $LOG_ROOT; pass the same flag if you submit this by hand.
#
# Freeze one training checkpoint into $CHECKPOINT_ROOT.  The copy is the only
# contact an evaluation ever has with a training run, which is what lets an arm
# be evaluated while it keeps training.
#
#   sbatch --output="$LOG_ROOT/%x_%j.log" slurm/snapshot.sbatch <arm> [final|latest|/path/to.ckpt]
#
# The work is a single copy, so run_snapshot.sh normally does it in the
# foreground; this template exists for the case where the checkpoint is large
# enough that copying it on a login node is antisocial.
set -uo pipefail
: "${PKG_ROOT:?source env.sh before submitting (it exports PKG_ROOT)}"
source "$PKG_ROOT/env.sh"
exec bash "$PKG_ROOT/run_snapshot.sh" "$@"
