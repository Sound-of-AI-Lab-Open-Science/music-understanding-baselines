# midi-rae-jepa Evaluation Protocol

General, checkpoint-agnostic procedure for evaluating any midi-rae-jepa
`SwinEncoder` checkpoint via [BenchMIR](https://github.com/inesbroto/BenchMIR)
and producing a bar chart in the same shape as `yuhang one model eval.png`
(dataset/task on the x-axis, score on the y-axis, one model's filtered-corpus
results only — no filtered-vs-unfiltered comparison).

This was developed and validated end-to-end on the 5-epoch backup checkpoint
(`SwinEncoder_musescore-big-backup_best.pt`); everything below is written to
apply unchanged to any other midi-rae-jepa checkpoint (e.g. the 16-epoch main
run) — the only thing that changes per checkpoint is the `checkpoint_path`
and `model_id` prefix in the config.

**Comparability across environments**: a direct-CLI/AWS counterpart to this
doc exists at `eval_protocol_aws.md`, for evaluating a checkpoint on a
machine without SLURM. Scores from the two are only meaningfully comparable
if both pin the **same BenchMIR commit** and **same torch/torchvision
versions** — a different commit or dependency version can silently change
scores even with an identical config (different probe init, different
numerics, or an upstream code change to a task/metric). Pin both explicitly
(§1) rather than just "latest" when comparability across a set of
checkpoints/environments matters.

---

## 0. What "the goal" actually is

Produce one PNG: dataset/task families on the x-axis (CIPI difficulty, EMOPIA
emotion, TopMAGD genre, POP909 chord/root/key, Composer/Humdrum), a score per
bar on the y-axis, for **one** trained checkpoint, evaluated on the
**filtered** training corpus only (MuseScore-big-MIDI filtered set — not a
filtered-vs-unfiltered comparison).

---

## 1. One-time environment setup (skip if already done)

BenchMIR needs newer/different dependency versions (torch>=2.13, specific
transformers pin) than the midi-rae training env, so it gets its **own**
conda env rather than reusing the training one.

```bash
# Clone BenchMIR (private repo -- SSH, not HTTPS, since it 404s unauthenticated)
cd /home/abahuguna/soundofai/eval
git clone git@github.com:inesbroto/BenchMIR.git
cd BenchMIR
# Pin the exact commit every checkpoint evaluated so far has used -- for
# scores to be comparable across checkpoints/environments (see note above),
# don't just take whatever's on main at clone time.
git checkout 87349a4159545c7cce45151335eef62e5dafd61e

# Dedicated env
source /gpfs/home/abahuguna/miniforge3/bin/activate
conda create -y -p /home/abahuguna/soundofai/eval/BenchMIR/.condaenv python=3.12
conda activate /home/abahuguna/soundofai/eval/BenchMIR/.condaenv

cd /home/abahuguna/soundofai/eval/BenchMIR
pip install -e ".[dev]" -q

# torch resolves to a cu13x build by default -- this cluster's GPU driver
# (570.x) only supports up to CUDA 12.8, so it must be downgraded or CUDA
# init silently reports cuda.is_available()==False. Same fix needed in the
# midi-rae training env; see that repo's docs_steps.log.
# Pinned to exact versions (not just the cu124 index) for comparability --
# a different resolved torch/torchvision patch version is a second silent
# way scores can drift between runs beyond just the BenchMIR commit.
pip uninstall -y torch torchvision -q
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 -q

# midi_rae itself (no-deps: BenchMIR doesn't need hydra/wandb/lpips/plotly,
# only midi_rae.swin.SwinEncoder + midi_rae.utils.load_checkpoint)
pip install -e /home/abahuguna/soundofai/midi-rae -q --no-deps
pip install timm pretty_midi omegaconf -q
```

Verify:
```bash
python3 -c "
from midi_rae.swin import SwinEncoder
from midi_rae.utils import load_checkpoint
import pretty_midi, torch
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
"
```

## 2. One-time data pull (skip if `experiments/data/raw/` already populated)

Two of the five families have no automatic downloader:

```bash
cd /home/abahuguna/soundofai/eval/BenchMIR
python3 -m benchmir.cli.main pull-data \
  --cipi-targz-path /home/abahuguna/soundofai/eval/difficulty_cipi.tar.gz \
  --humdrum-targz-path /home/abahuguna/soundofai/eval/composer_humdrum.tar.gz \
  --datasets cipi,humdrum,emopia,pop909,topmagd
```

Where the tarballs came from: manually sourced by the user (Google Drive
export), unzipped from `drive-download-*.zip` into
`difficulty_cipi.tar.gz` / `composer_humdrum.tar.gz` under
`/home/abahuguna/soundofai/eval/`. If evaluating on a machine that doesn't
have these already, they need to be re-obtained the same way — there's no
public mirror BenchMIR knows how to pull them from automatically.

EMOPIA, POP909, and topMAGD (LMD-matched + MSD genre assignment) pull
automatically. topMAGD is the large one (~5.5GB) and can take several
minutes.

Validate the pull (catches path/version issues before committing to a GPU
run — cheap, run on the login node):
```bash
python3 -c "
from benchmir.eval.datasets.corpora.cipi_eval_corpus import CIPIDifficultyEstimationEvalDataset
from benchmir.eval.datasets.corpora.emopia_eval_corpus import EMOPIAEmotionClassificationEvalDataset
from benchmir.eval.datasets.corpora.humdrum_eval_corpus import HumdrumComposerClassificationEvalDataset
from benchmir.eval.datasets.corpora.pop909cl_eval_corpus import POP909ChordEstimationEvalDataset
from benchmir.eval.datasets.corpora.top_magd_eval_corpus import MSDTopMAGDGenreClassificationEvalDataset

print('CIPI:', len(CIPIDifficultyEstimationEvalDataset(root_dir='experiments/data/raw/cipi/', target_class='henle', symbolic_format='midi')))
print('EMOPIA:', len(EMOPIAEmotionClassificationEvalDataset(root_dir='experiments/data/raw/emopia/', target_class='4Q')))
print('Humdrum:', len(HumdrumComposerClassificationEvalDataset(root_dir='experiments/data/raw/humdrum/', target_class='composer', symbolic_format='midi')))
print('POP909:', len(POP909ChordEstimationEvalDataset(root_dir='experiments/data/raw/POP909cl/', n_folds=5, val_ratio=0.15)))
print('TopMAGD:', len(MSDTopMAGDGenreClassificationEvalDataset(root_dir='experiments/data/raw/LMDMatched/', use_all_matches=False, n_folds=5, val_ratio=0.15)))
"
```
Expected (snapshot, may drift slightly): CIPI 592, EMOPIA 1071, Humdrum 1968,
POP909 909, TopMAGD ~10,282.

## 3. The model adapter (already written, reused as-is per checkpoint)

`BenchMIR/src/benchmir/models/midi_rae_jepa.py` (`MidiRaeJepaEmbeddingModel`,
registered in `models/registry.py`). This is checkpoint-agnostic — it takes
`checkpoint_path` as a constructor kwarg, so **do not edit this file to
evaluate a different checkpoint**; only the config changes (§5).

**Caveat, easy to miss (2026-09-15): the Swin architecture is hardcoded, not
inferred from the checkpoint.** The adapter constructs
`SwinEncoder(img_height=128, img_width=128, patch_h=4, patch_w=4, embed_dim=8,
depths=[2,2,2,6,2,1], num_heads=[2,2,2,4,8,16], window_size=4, mlp_ratio=4.0,
drop_path_rate=0.0)` unconditionally, then calls `load_checkpoint(...,
strict=False)`. This matches every checkpoint evaluated so far (backup,
main16 -- both trained from `config_swin.yaml`'s architecture). If you're
evaluating a checkpoint trained with **any different architecture**
(different `embed_dim`, `depths`, `num_heads`, patch size, or window size --
e.g. a different config file, or a checkpoint from someone else's training
run with unknown hyperparameters), `strict=False` will silently load only the
keys that happen to match in shape and skip the rest **without erroring** --
you'd get a model that runs and produces numbers, just wrong/undertrained
ones, with no crash to tip you off.

Before evaluating an externally-trained checkpoint, verify the architecture
matches first:
```bash
python3 -c "
import torch
sd = torch.load('/path/to/checkpoint.pt', map_location='cpu', weights_only=False)['model_state_dict']
# spot-check a few shapes against the hardcoded architecture above, e.g.:
for k in sd:
    if 'patch_embed' in k or 'stages.0' in k: print(k, sd[k].shape)
"
```
If the checkpoint's config (or training logs, if you have them) specifies
different hyperparameters than the block above, you'll need to either
parametrize the adapter's `SwinEncoder(...)` call (via a new constructor
kwarg, not by hand-editing the hardcoded literals for a one-off) or confirm
they in fact do match before trusting the results.

What it does:
- Converts a raw MIDI file to a tempo-normalized 32nd-note binary piano roll,
  identical recipe to midi-rae's own training-time preprocessing
  (`midi-rae/scripts/midi_to_pianoroll.py`) — 8 steps/beat, capped at 4096
  columns (~512 beats) so one pathologically long/dense file can't blow up
  per-file runtime.
- Slides a 128x128 window across the roll (default stride 64, i.e. 50%
  overlap) and runs every crop through the frozen `SwinEncoder` in **one
  batched forward pass** (not a per-crop loop — this was a real bug caught
  during development: unbatched CPU inference took minutes per song).
- `level` kwarg selects what's returned: `"concat"` (all 6 hierarchy levels
  mean-pooled-then-concatenated, 504-dim) or `0`-`5` (a single level,
  `0`=coarsest L0, `5`=finest L5 — matches
  `HierarchicalPatchState.levels` ordering, coarsest first).
- Song-level (`run`/`run_batch`, feeds CIPI/EMOPIA/TopMAGD/Humdrum):
  mean-pools every crop's embedding into one fixed vector per song.
- Frame-level (`run_frames`, feeds POP909 chord/root/key): keeps every
  patch of the chosen level as its own timestamped frame (start/end seconds
  derived from the song's tempo and the patch's column position). For
  `level="concat"`, frames come from the finest level (L5) since coarse
  levels' patches span multiple bars and aren't meaningful as time-aligned
  frames.
- Corrupted/unparseable MIDI files are caught and degrade to a zero
  embedding (song-level) or zero frames (frame-level) with a printed
  warning, rather than crashing the whole run — this **was** a crash bug
  (a single malformed file killed a job that had already run 20+ minutes)
  before the fix; if re-implementing this from scratch, don't skip this.

## 4. Smoke-test before committing to the full run

Always validate the full pipeline (extraction -> probe -> report) on one
small task with one model variant, interactively, before submitting the
expensive full job. Catches config/registry typos and device bugs for the
cost of ~1 minute instead of discovering them after an hour of GPU time.

```bash
srun -p high --gres=gpu:tesla:1 --cpus-per-task=8 --mem=16G --time=00:30:00 bash -c "
source /gpfs/home/abahuguna/miniforge3/bin/activate /home/abahuguna/soundofai/eval/BenchMIR/.condaenv
cd /home/abahuguna/soundofai/eval/BenchMIR
python3 -m benchmir.cli.main run --config configs/_smoketest.yaml
"
```
(`_smoketest.yaml`: one model entry pointing at the checkpoint under test,
`level: concat`, just the CIPI task, 5 folds.) Sanity checks:
- Extraction finishes in well under a minute for 592 songs.
- Probe fit+eval scores print and look non-degenerate (not exactly chance,
  not NaN).
Delete/overwrite `runs/smoketest/` afterward — it's disposable.

## 5. The real config: one model, all 7 hierarchy-level variants, all 7 tasks

Copy `configs/midirae_backup_eval_v1.yaml` as a starting point (or
regenerate it following the shape below) and do a global find/replace of
the checkpoint path and the `model_id` prefix:

- `checkpoint_path`: point at the new checkpoint's `.pt` file, e.g.
  `/home/abahuguna/soundofai/midi-rae/checkpoints/SwinEncoder_<TAG>_best.pt`
- `model_id` prefix: rename `midirae-backup-*` to something identifying the
  new checkpoint, e.g. `midirae-main16-*` — **must** stay unique per
  checkpoint, since BenchMIR's embedding cache is keyed by `model_id`, not
  checkpoint path; reusing an old `model_id` with a new checkpoint silently
  serves stale cached embeddings.
- `device: "cuda"` on every model entry (not `"cpu"` — see §6 on why).
- `run_id` / `output_dir` under `run:`: give the new run its own directory
  so it doesn't collide with a prior checkpoint's report.

14 model entries total: `{concat, L0, L1, L2, L3, L4, L5}` x
`{song-level, frame-level ("-frames" suffix)}`. 7 `eval_tasks` entries:
`cipi`, `emopia`, `topmagd`, `humdrum` (each listing all 7 song-level
`model_ids`), `pop909_chord`, `pop909_root`, `pop909_key` (each listing all
7 frame-level `model_ids`).

Why all 7 levels rather than just one: the papers show different hierarchy
levels win at different tasks (coarse levels for phrase/global structure,
fine levels for note-level/harmonic detail) — evaluating only one level
would silently hide that structure. Reducing this to a single bar per task
for the final figure is a separate, later step (§8).

## 6. Compute: check GPU availability before choosing a partition

CPU inference works but is **not viable at this corpus scale** — TopMAGD
alone is ~10,000 songs. Always target GPU. Two options exist on this
cluster; check actual free capacity before picking one (do not assume):

```bash
squeue -w node027 -o "%.10i %.10u %.10P %R"           # impa (unlimited walltime, but often fully occupied)
for n in node019 node020 node021 node022 node023; do  # high partition, tesla GPUs, 14-day cap
  echo "--- $n ---"; scontrol show node $n | grep -E "Gres=|AllocTRES"
done
```
If `impa`'s single node (4x L40S) is fully allocated (common if a training
job is also running there), fall back to `--partition high --gres=gpu:tesla:1`
— slower hardware but a 14-day cap is still effectively unlimited for this
job, and there are usually several free Tesla GPUs. Do not hardcode a
specific node; let the scheduler place it (`--gres=gpu:tesla:1`, no `-w`).

sbatch template (`slurm/02_run_eval.sbatch`):
```bash
#!/bin/bash
#SBATCH --job-name=benchmir-eval
#SBATCH --partition=high
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14-00:00:00
#SBATCH -o /gpfs/home/abahuguna/soundofai/eval/BenchMIR/slurm_logs/%x-%j.out
#SBATCH -e /gpfs/home/abahuguna/soundofai/eval/BenchMIR/slurm_logs/%x-%j.err

set -euo pipefail
cd /gpfs/home/abahuguna/soundofai/eval/BenchMIR
PY=/gpfs/home/abahuguna/soundofai/eval/BenchMIR/.condaenv/bin/python3
$PY -c "import torch; print('torch OK', torch.__version__, 'cuda:', torch.cuda.is_available())"
$PY -m benchmir.cli.main run --config configs/<YOUR_CONFIG>.yaml
echo "BENCHMIR_EVAL_DONE"
```

Submit and watch:
```bash
sbatch slurm/02_run_eval.sbatch
# progress: grep -E "extracted [0-9]+ embedding|extracted [0-9]+ frame|Traceback" slurm_logs/benchmir-eval-<jobid>.err
```

Known-good extraction throughput on a Tesla GPU (for sizing expectations on
a fresh run): CIPI (592 songs) ~42s, EMOPIA (1071) ~24s, Humdrum (1968)
~3.4min, TopMAGD (10,282) ~28min. These repeat **per model variant** — no
cache sharing across different `model_id`s — so budget roughly 30-35 min
per song-level variant x 7 variants for the song-level tasks alone (~3.5-4
hours), plus POP909's 3 frame-level tasks x 7 variants on top (timing not
yet characterized at the time of writing — extraction should be comparably
fast since it's the same crops/forward-passes as song-level, just returned
ungrouped; probe training time on the much larger frame-count datasets is
the open unknown).

## 7. Known bugs already fixed upstream in this BenchMIR checkout

Both were hit and fixed once already — should not recur for a new checkpoint
using the same config/environment, but documented here in case the BenchMIR
checkout gets refreshed from upstream and reintroduces them:

- **`models/midi_rae_jepa.py`**: originally looped `run()` per-crop
  (unbatched) with no length cap -> minutes-long hangs on long songs. Fixed
  with batched crop inference (`_crop_batches`, max 64 crops/batch) and a
  4096-column cap matching training preprocessing.
- **`eval/tasks/corpora/topMAGD_genre_task.py`** `build_loss_fn()`:
  `nn.CrossEntropyLoss(weight=weights)` — `weights` was never moved to
  `cuda`, crashing with a device-mismatch `RuntimeError` the moment a probe
  actually ran on GPU (which is every real run, since `GradientProbe`
  defaults to `cuda` when available, independent of the embedding model's
  own `device` setting). Fixed: `weights.to("cuda" if torch.cuda.is_available() else "cpu")`.

## 8. Turning the report into the final PNG (not yet scripted — do this once results land)

BenchMIR writes a JSON report under `<output_dir>/` (per `report: {format:
"json"}` in the config) with per-(model, task, fold) scores. To match
`yuhang one model eval.png`'s shape:

1. Pick the metric that gives one number per task (mirror the reference
   image's caption — it uses task-appropriate primary metrics: accuracy for
   CIPI/EMOPIA/TopMAGD/Humdrum, `chord_root`/`key_weighted_score` for
   POP909 root/key, plain accuracy for POP909 chord). Average across folds
   where the task is folded.
2. Collapse the 7 hierarchy-level variants to one number per task. Two
   choices, pick one and say so in the figure's caption/subtitle (mirroring
   how the reference figure's own caption spells out its methodology in
   detail):
   - **Best-level-per-task** (matches the papers' own STORMBIRD Table 1
     convention: "each cell reports the best value across levels, winning
     level noted in small type") — most directly comparable to the papers.
   - **Grouped bars**, one sub-bar per level per task — more information,
     less directly comparable to the single-bar reference figure.
3. One bar per task family (CIPI difficulty, EMOPIA emotion, TopMAGD genre,
   POP909 chord, POP909 root, POP909 key, Composer Humdrum — 7 bars, same
   x-axis order as the reference), y-axis = score, single series (this
   checkpoint only — no filtered/unfiltered split, per the original
   instruction). Include a chance-level dashed reference line per task if
   reproducing the reference figure's style exactly (it uses uniform-chance
   baselines, e.g. 1/num_classes).
4. Load the `dataviz` skill before writing the actual plotting code —
   covers color/style conventions this repo/session already follows
   elsewhere.

## 9. Repeating for a new checkpoint — condensed checklist

1. Confirm §1/§2 (env + data) are already done — skip if so, they're
   checkpoint-independent.
2. Copy the config, swap `checkpoint_path` and the `model_id` prefix (§5).
   New `run_id`/`output_dir`.
3. Smoke-test (§4) with the new checkpoint before the full run.
4. Check GPU availability, submit via SLURM (§6).
5. Once done, produce the figure (§8) — same script/process as any other
   checkpoint, just pointed at the new run's JSON report.

## 10. Bug found post-writing (2026-09-15): frame pooling across pitch

`run_frames()` originally emitted one frame per (pitch-row, time-column)
patch, not one per time-column. Since chord/root/key labels are purely
time-aligned (one label per time window, independent of pitch), this
produced `grid`x too many frames — e.g. **36.7 million** rows for POP909's
909 songs at the finest level (grid=32), instead of the expected ~1M, all
sharing duplicate timestamps. Caught by watching `extracted N embedding
rows` in the job log and noticing an implausible row count.

Fixed in `midi_rae_jepa.py`'s `run_frames()`: group patches by their time
column and mean-pool the embedding across the pitch (row) dimension before
emitting one frame per column. Validated: a file that produced 2,432 frames
pre-fix (38 crops x 8x8 grid at L3) produced exactly 304 post-fix (38 crops
x 8 columns) — matches `n_crops x grid` exactly, confirming the fix removes
precisely the row-duplication and nothing else.

**If evaluating a checkpoint and this file has since been reverted/refetched
from a stale copy, re-check `run_frames()` pools over pos[:,0] (rows) before
emitting -- do not assume the fix is permanently present.** Also: if a prior
run's `runs/<run_id>/cache/embeddings/*pop909*` files exist from before this
fix, delete them before rerunning -- the cache key doesn't encode this kind
of internal semantic change, so a stale cache will silently serve the
bugged (36x-inflated, row-duplicated) frames forever.
