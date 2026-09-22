# MIDI-RAE-JEPA arm

This directory holds what's needed to **re-train** the MIDI-RAE-JEPA encoder/decoder
pair whose numbers appear in the paper's results table, and to **re-run the eval**
that produced them. It does not include checkpoints, training data, or exploratory
configs from other runs — see "What's deliberately left out" below.

Upstream project: https://github.com/drscotthawley/midi-rae (Apache-2.0, see
`LICENSE` / `UPSTREAM_README.md`). This code is a modified copy, not a submodule —
see "Local modifications" below for what changed from upstream.

**Not wired into `pretrain/baselines/train.py`.** The three arms next to this one
(`jepa`, `musicbert`, `musetok`) share one CLI, dispatched through
`pretrain/baselines/models/registry.py` into Lightning `DataModule`/
`LightningModule` pairs over an Octuple or REMI+ token cache. MIDI-RAE-JEPA is a
different pipeline end to end: Hydra configs (not the registry's YAML shape),
its own `midi_rae.train_enc` / `midi_rae.train_dec` entrypoints (no Lightning),
piano-roll image shards (not Octuple/REMI+ tokens), and its own conda env, not
one of `$JEPA_ENV` / `$MUSICBERT_ENV` / `$MUSETOK_ENV`. It sits alongside the
other arms for discoverability, and runs standalone via the steps below —
`run_pretrain.sh <arm>` does not know about it.

## What's here

- `midi_rae/` — the full upstream package, plus local modifications (below).
- `configs/config_swin_full_backup.yaml` — **the only training config kept**. This
  is the 5-epoch "backup" run that ended up producing the numbers reported in the
  paper (it trained cleanly; a longer 16-epoch "main" run hit two mid-training
  instabilities and was not used for the final table). It's the config as actually
  run, including its original absolute cluster paths (`/gpfs/home/abahuguna/...`)
  — update `data.path`, `preencode.output_dir`, and `encoder_ckpt` for your own
  environment before use.
- `scripts/midi_to_pianoroll.py` — converts raw MIDI to tempo-normalized,
  32nd-note-quantized piano-roll shards. Needs a CSV of filtered filenames as
  input (see "What's deliberately left out").
- `slurm/` — the three SLURM job scripts that actually produced the reported
  checkpoint: preprocess → train encoder → train decoder (backup variants only).

## Local modifications to upstream midi-rae

- `midi_rae/data.py` — added `ShardedAnchorDataset`/`ShardedTripletDataset` for
  lazy, sharded loading of a ~1M-file corpus (upstream loads everything into RAM).
  Hand-written, not covered by upstream's nbdev regeneration.
- `midi_rae/train_enc.py` — added the `cfg.data.format == "sharded"` branch, and
  replaced a hardcoded EMA eta hard-jump (`if epoch==44: ema_encoder.eta=0.96`)
  with a configurable, gradual log-space ramp (`ema_eta_switch_epoch`,
  `ema_eta_ramp_epochs`, `ema_eta_after` in the config).
- `midi_rae/train_dec.py` — parallel sharded-dataset branch.
- `midi_rae/utils.py` — fixed a `save_checkpoint()` bug where a default
  `save_every=25` silently skipped all checkpoints on runs shorter than 25 epochs.

## How to re-train (backup/5-epoch arm)

1. Obtain your own filtered MuseScore-big-MIDI file list (a CSV with at least a
   filename column) and point `slurm/01_preprocess.sbatch` at your local copy of
   the raw MIDI corpus.
2. Update the absolute paths in `configs/config_swin_full_backup.yaml` for your
   cluster/environment.
3. `pip install -e .` from this directory (installs `midi_rae` and its
   dependencies, per `pyproject.toml`).
4. Submit `slurm/01_preprocess.sbatch`, then `slurm/02b_train_enc_backup.sbatch`,
   then `slurm/03b_train_dec_backup.sbatch` (encoder before decoder — the decoder
   trains against the frozen encoder's embeddings).

## What's deliberately left out (no dataset redistribution)

- Raw MIDI files, the filtered-file CSV, POP909/CIPI/EMOPIA/TopMAGD/Humdrum eval
  data, and any preprocessed piano-roll shards.
- All checkpoints (`checkpoints/`, `saved/`) and training logs/W&B runs.
- The 16-epoch "main" config and its resume/diagnostic variants
  (`config_swin_full.yaml`, epoch-6 checkpoint config, etc.) — not the arm whose
  results are reported, so not reproduced here. Ask if you want the full
  training-instability history preserved too; it's documented separately.
