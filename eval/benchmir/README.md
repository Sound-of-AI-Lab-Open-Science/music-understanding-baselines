# MIDI-RAE-JEPA eval (BenchMIR)

The eval numbers reported for MIDI-RAE-JEPA were produced with
[BenchMIR](https://github.com/inesbroto/BenchMIR), following `eval_protocol.md`
in this directory. BenchMIR is **not vendored here** (no dataset redistribution,
and its own eval corpora — POP909, CIPI, EMOPIA, TopMAGD, Humdrum — are
third-party data you need to obtain yourself per BenchMIR's own instructions).

## What's here

- `eval_protocol.md` — the evaluation protocol actually followed.
- `configs/midirae_backup_eval_v1.yaml` — the BenchMIR run config for the
  5-epoch "backup" checkpoint, i.e. the one whose numbers are in the paper.
  Other configs used during development (a 16-epoch "main" eval, a
  pre-instability epoch-6 diagnostic, a pitch-concat variant, a downbeat-align
  variant) are not included — they were exploratory, not part of the reported
  result. Ask if you want them added.
- `patches/` — **required** to actually run the config above. The BenchMIR fork
  used (`inesbroto/BenchMIR`) did not have MIDI-RAE-JEPA model support; these
  are the local additions/fixes made on top of it:
  - `midi_rae_jepa.py` — new file, drop into `src/benchmir/models/` in your
    BenchMIR checkout. Defines `MidiRaeJepaEmbeddingModel`, which loads a
    midi-rae encoder checkpoint and extracts embeddings at a chosen hierarchy
    level (`concat` or `L0`–`L5`).
  - `registry.py.patch` — registers the model class above.
  - `topMAGD_genre_task.py.patch` — a fix to the TopMAGD genre eval task.

  **Known gap**: these patches were never upstreamed to `inesbroto/BenchMIR`, and
  this branch does not pin or vendor that fork. To actually reproduce the eval,
  apply the three patches above to a checkout of `inesbroto/BenchMIR` yourself.

## How to reproduce

1. Clone `inesbroto/BenchMIR` and apply the patches in `patches/` (copy
   `midi_rae_jepa.py` in, `git apply` the two `.patch` files).
2. Obtain the eval datasets BenchMIR expects (see its own docs) — not included
   here.
3. Point `configs/midirae_backup_eval_v1.yaml`'s checkpoint path at the encoder
   produced by `midi-rae/slurm/02b_train_enc_backup.sbatch` in this repo.
4. Run BenchMIR with that config, per `eval_protocol.md`.
