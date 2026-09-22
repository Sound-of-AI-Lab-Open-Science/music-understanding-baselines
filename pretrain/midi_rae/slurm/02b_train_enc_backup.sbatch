#!/bin/bash
#SBATCH --job-name=midirae-enc-backup
#SBATCH --partition=high
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14-00:00:00
#SBATCH -o /gpfs/home/abahuguna/soundofai/midi-rae/slurm_logs/%x-%j.out
#SBATCH -e /gpfs/home/abahuguna/soundofai/midi-rae/slurm_logs/%x-%j.err

set -euo pipefail
cd /gpfs/home/abahuguna/soundofai/midi-rae
PY=/gpfs/home/abahuguna/soundofai/midi-rae/.condaenv/bin/python3
$PY -c "import torch; print('torch OK', torch.__version__, 'cuda:', torch.cuda.is_available())"

nvidia-smi

$PY -m midi_rae.train_enc --config-name config_swin_full_backup ++tag=musescore-big-backup \
  ++hydra.run.dir=. ++hydra.job.chdir=False ++hydra.output_subdir=null

echo "TRAIN_ENC_BACKUP_DONE"
