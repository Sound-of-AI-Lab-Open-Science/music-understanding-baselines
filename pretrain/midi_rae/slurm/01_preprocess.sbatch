#!/bin/bash
#SBATCH --job-name=midirae-preproc
#SBATCH --partition=high-cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH -o /gpfs/home/abahuguna/soundofai/midi-rae/slurm_logs/%x-%j.out
#SBATCH -e /gpfs/home/abahuguna/soundofai/midi-rae/slurm_logs/%x-%j.err

set -euo pipefail
cd /gpfs/home/abahuguna/soundofai/midi-rae
PY=/gpfs/home/abahuguna/soundofai/midi-rae/.condaenv/bin/python3
$PY -c "import torch; print('torch OK', torch.__version__)"

$PY scripts/midi_to_pianoroll.py \
  --csv dataset_stats_filtered.csv \
  --src-dir /projects/mtg/projects/andrea/MuseScore-big-MIDI \
  --out-dir /gpfs/home/abahuguna/soundofai/midi-rae/data/pianorolls \
  --steps-per-beat 8 --max-len 4096 --shard-size 2000 --val-frac 0.02 --workers ${SLURM_CPUS_PER_TASK}

echo "PREPROCESS_DONE"
