# JEPA-reproduce

A from-scratch reproduction of **Music-JEPA** — a joint-embedding predictive
architecture for symbolic music — together with two baselines pre-trained on the
same corpus, the same split and the same budget, and evaluated on the same
probes.

Three models are pre-trained here, four arms in total:

| arm | model | tokenization | what it is for |
|---|---|---|---|
| `jepa_paper` | Music-JEPA | OctupleMIDI | the paper-true recipe: 16 layers, absolute **and** relative position encoding, unweighted VICReg |
| `jepa_champA` | Music-JEPA | OctupleMIDI | a smaller, more strongly regularised control — a JEPA can reach a low loss while its representation collapses, and one run cannot tell you which happened |
| `musetok` | MuseTok | REMI+ | residual-VQ tokenizer, retrained with a corpus-derived vocabulary |
| `musicbert` | MusicBERT | OctupleMIDI | masked-LM baseline on the official recipe's epoch budget |

They are then frozen and probed on symbolic-MIR tasks — piano difficulty,
emotion, genre, chord, chord root, key, composer — through an external
evaluation library, so that no model is scored by code that was written for it.

**Attribution**: the evaluation harness is BenchMIR (Broto Clemente), used as a
pinned git submodule. See `THIRD_PARTY_NOTICES.md` for it and for everything
else this package depends on but does not redistribute.

---

## Requirements

* A Linux cluster with **SLURM**. Every long step is submitted as a job; the
  templates in `slurm/` carry no site-specific flags and read `env.sh`.
* **conda** (any distribution), and disk for three environments. They exist
  because their torch builds are mutually exclusive — see *Why three
  environments* below.
* **One GPU** per pre-training arm. Everything after pre-training is CPU work.
  The MuseTok arm is the one with a floor: at its recipe's micro-batch 16 over
  `dec_seqlen` 1280 it needs **more than 16 GB** — measured, it fills a 15 GB
  card and dies — so pin it with `JR_SLURM_GRES` / `JR_SLURM_EXCLUDE`, or halve
  `data.batch_size` and double `trainer.accumulate_grad_batches` in
  `musetok.yaml`, which leaves the effective batch at 64. The three other arms
  fit a 15 GB card as they stand.
* Storage: roughly 80 GB of token caches for a 1.7 M-file corpus, plus the
  checkpoints. The union caches are symlinks and cost almost nothing.
* The corpora, which are **not** included. `./check_data.sh` tells you exactly
  what is missing and where it is looked for.

---

## Layout

```
.
├── env.sh                  source this; it is the only place a path is decided
├── paths.yaml              the defaults env.sh and pkgpaths.py both read
├── pkgpaths.py             the same resolution, for Python
├── setup.sh                submodule + three conda envs + editable install
├── check_data.sh           verify the corpora layout before submitting anything
├── run_tokenize.sh         1. tokenize the corpus            (SLURM array)
├── run_union.sh            2. split, union caches, vocabulary (foreground)
├── run_pretrain.sh         3. pre-train one arm               (SLURM, chained)
├── run_snapshot.sh         4. freeze a checkpoint for evaluation
├── run_encode.sh           5. fill the per-bar embedding cache (SLURM array)
├── run_eval.sh             6. encode + fit the probes         (SLURM arrays)
├── run_report.sh           7. merge report.json -> a results table
├── run_smoke.sh            prove the pipeline before committing a GPU week
├── lib/grids.sh            which evaluation grid means what
├── lib/smoke_corpus.py     cuts the tiny corpora --end-to-end runs on
├── slurm/                  five generic templates: tokenize, pretrain,
│                           snapshot, encode, eval
├── envs/                   the three environments, as exported specs
├── pretrain/
│   ├── src/                the model package: jepa.py, musicbert.py, the
│   │                       Octuple codec, VICReg, EMA, relative attention
│   ├── baselines/          the training scaffold behind one CLI
│   │   ├── train.py        discover -> filter -> cache -> build -> fit
│   │   ├── configs/        the four recipes, one per arm
│   │   ├── cache_tools/    union merge, vocabulary, keep-list
│   │   ├── data/           Octuple and REMI+ caches, datamodules
│   │   └── models/         the registry that wires config -> Lightning module
│   └── scripts/            make_musescore_split.py (the content-hash split) and
│                           the three model drivers train.py builds through:
│                           train_jepa.py, train_mlm.py, preprocess_lmd.py
├── eval/benchmir/
│   ├── configs/            the evaluation recipes (models x datasets x tasks)
│   ├── scripts_ours/       job generators, the cache-warming driver, report makers
│   └── scripts_topmagd/    the three-split-strategy genre study
└── third_party/
    ├── BenchMIR/           submodule: the evaluation library, at a pinned commit
    ├── MuseTok/            NOT vendored; setup.sh --with-musetok clones it
    ├── licenses/           upstream licence texts we are required to carry
    └── patches/            what our submodule branch changes in an upstream file
```

---

## Setup

```bash
git clone <this repository>      # plain clone; see the note below
cd <this repository>

source env.sh            # defines PKG_ROOT and every other path
./setup.sh               # submodule, three conda envs, editable install
./check_data.sh          # says what is missing, and where it was looked for
./run_smoke.sh           # proves the environments and every recipe resolve
```

**Clone without `--recurse-submodules`; let `./setup.sh` try the submodule.**
The evaluation library is a submodule pointing at a fork of a third-party
repository, and whether that fork is published is the BenchMIR authors'
decision, not this package's (`THIRD_PARTY_NOTICES.md` records this), so the
URL may not resolve for you. `git clone --recurse-submodules` against a
submodule it cannot read prompts for credentials and then **exits non-zero** —
the superproject lands, but the command fails. `./setup.sh` runs the same fetch
with prompting disabled, so it cannot hang: it warns, carries on, and leaves
you a working pre-training half.

Everything there — `run_tokenize.sh`, `run_union.sh`, `run_pretrain.sh`,
`run_snapshot.sh` — works without the submodule, and `./run_smoke.sh` reports
the evaluation checks as *skipped* rather than failed. `run_eval.sh` and
`run_report.sh` are the two that need it; if you get access to the fork later,
`git submodule update --init` at any time completes the setup.

`./setup.sh --with-musetok` additionally clones the upstream MuseTok checkout,
at its pinned commit, into `third_party/MuseTok`. It is a separate flag because
upstream publishes no licence file; read the MuseTok section of
`THIRD_PARTY_NOTICES.md` before running it. Only the `musetok` arm needs it.

### The one line you have to set

```bash
export DATA_ROOT=/where/the/corpora/live      # everything else defaults off this
```

`DATA_ROOT/musescore` is the pre-training corpus (searched recursively, any
nesting) and `DATA_ROOT/benchmir` holds the evaluation corpora in the layout
`check_data.sh` prints. Any of the derived paths can be overridden on its own —
`MIDI_DIR`, `BENCHMIR_DATA_ROOT`, `WORK_ROOT`, `CHECKPOINT_ROOT`. Cluster
policy goes in `JR_SLURM_PARTITION`, `JR_SLURM_ACCOUNT`, `JR_SLURM_QOS`,
`JR_SLURM_EXCLUDE` and `JR_SLURM_GRES` (the `JR_` prefix keeps them out of
Slurm's own reserved `SLURM_*` namespace); an empty one contributes no flag at
all, so the package runs on a cluster with no partition policy without edits.
Defaults live in `paths.yaml`; an exported variable always wins, so relocating
the package never requires editing a file.

### Why three environments

Not tidiness — **their torch builds are mutually exclusive**. The MuseTok
checkpoints need `miditoolkit==1.0.0` (1.0.1 changed tick handling and silently
produces a different REMI+ encoding from the same MIDI), and the evaluation
library pins a torch with no CUDA wheel for many drivers. So each model adapter
runs its model in a **subprocess** under its own interpreter and talks to it
over a framed stdin/stdout protocol. That is also why the probe jobs are CPU
jobs and why embedding extraction is pushed out into the two GPU-capable
environments.

---

## The pipeline, in order

```
  raw MIDI corpus
        │
  [1]   ./run_tokenize.sh octuple          one array task per shard, CPU, ~20 min/shard
        ./run_tokenize.sh remi             ...REMI+ is ~10x slower; both are resumable
        │
  [2]   ./run_union.sh                     split on CONTENT, union the shard caches,
        │                                  derive the MuseTok vocabulary
        │
  [3]   ./run_pretrain.sh <arm>            one GPU job per arm; self-chaining, so a
        │                                  walltime cap cannot end a run
        │
  [4]   ./run_snapshot.sh <arm> latest     freeze the checkpoint; training may continue
        │
  [5]   ./run_eval.sh <eval-arm>           encode to the per-bar cache, then fit probes
        │
  [6]   ./run_report.sh                    merge report.json -> CONSOLIDATED.md
```

The **only** shared state between the evaluation half and the training half is
the frozen checkpoint snapshot. An evaluation never reads a live
`runs/*/checkpoints/` directory.

### 1. Tokenize

```bash
./run_tokenize.sh octuple      # Music-JEPA and MusicBERT share this cache
./run_tokenize.sh remi         # MuseTok
./run_tokenize.sh octuple 5,7,12   # re-run three shards that failed
```

One array task per immediate subdirectory of the corpus, sorted, so the mapping
from array index to shard is stable and a single shard can be resubmitted.

**Do not put a `filter:` block in a recipe.** The scaffold hashes the file list
into the cache directory name, so a filter changes the hash and forces a full
re-tokenization. Corpus restriction happens in step 2, as a keep-list.

### 2. Split, union, vocabulary

```bash
./run_union.sh                                   # the whole corpus
./run_union.sh --keep-ids ids_keep.txt           # restricted to a keep-list
SPLIT_TAG=filtered ./run_union.sh --keep-ids ids_keep.txt   # as its own variant
```

Three things happen, and the order is not negotiable:

* **The split is keyed on file CONTENT, not on corpus id.** The same piece is
  uploaded many times under different ids; keying on id put 12.1 % of the
  held-out test set into train. The script must report *0 ids with no content
  hash* — anything else means tokenization is incomplete and the split would
  silently leak.
* **Held-out test ids never enter a union cache at all** — not merely excluded
  at sampling time, absent from the index.
* **The MuseTok vocabulary is derived from the corpus.** Upstream's released
  168-token dictionary comes from a piano corpus; on a broad corpus roughly a
  quarter of pieces carry at least one event it cannot express, and the dataset
  raises `KeyError` on the first one. Deriving it is also what the paper did.
  The consequence is stated loudly because it is easy to trip over: **token ids
  shift, so checkpoints from this pipeline are not vocabulary-compatible with
  the public MuseTok weights.**

A keep-list is one id per line; `pretrain/baselines/cache_tools/build_keep_list.py`
builds one from a quality CSV. Give each corpus variant its own `SPLIT_TAG`: the
merge *writes* `pieces_{train,val}.txt` into the split directory, so two
variants sharing one directory overwrite each other's piece lists.

### 3. Pre-train

```bash
./run_pretrain.sh jepa_paper
./run_pretrain.sh jepa_champA
./run_pretrain.sh musetok
./run_pretrain.sh musicbert
./run_pretrain.sh jepa_paper --smoke     # 20 files, 200 steps, same batch shape
```

Each job **self-chains**: before training it queues its own successor with
`--dependency=afterany` against the same output directory and `--resume`, so a
partition walltime cap cannot end a run without a continuation queued. The chain
stops itself when `checkpoints/final.ckpt` appears. `JR_CHAIN=0` submits a
single job.

Two mechanics worth knowing before editing anything mid-flight:

* The chain re-executes SLURM's **spool copy** of the template, so editing
  `slurm/pretrain.sbatch` cannot affect a running chain — but a fresh submit
  picks it up.
* The **recipe is re-read from the package at every segment**, so a config edit
  *does* reach the next segment. Do not edit a recipe while its chain is alive.

A low per-dimension variance at one step says nothing about collapse; only its
trajectory does. In a healthy run it dips and then rises monotonically — that is
VICReg's variance term working, not a model dying.

### 4. Snapshot

```bash
./run_snapshot.sh jepa_paper latest      # highest periodic checkpoint
./run_snapshot.sh musetok final          # written at max_steps
./run_snapshot.sh musicbert /path/to/a.ckpt
./run_snapshot.sh jepa_paper latest --smoke   # from runs/<arm>_smoke
```

Writes `<arm>_step<N>.ckpt` (immutable, and what the generated job configs
record) plus a `<arm>.ckpt` symlink to it (what the recipes name, so they never
need editing). Copy-then-rename, because the training jobs are live and a
checkpoint can be half-written under a reader.

`--smoke` reads `runs/<arm>_smoke` instead of `runs/<arm>`, which is where
`run_pretrain.sh <arm> --smoke` writes, and stamps the copy
`<arm>_smoke_step<N>.ckpt` — so thirty steps of proof can never be mistaken on
disk for a finished run.

### 5. Evaluate

```bash
./run_eval.sh jepa_musescore_paper
./run_eval.sh musetok_musescore cipi_difficulty emopia_emotion
GRID=composer ./run_eval.sh all
GRID=v1       ./run_eval.sh all      # the released-checkpoint reference grid
```

**Launch one arm at a time, the moment that arm stops training.** Waiting for
the whole grid puts the slowest run on every other run's critical path. Each arm
writes into its own jobs directory and a shared runs root, and `run_report.sh`
merges whatever has finished — it can be re-run later with more columns.

`GRID` selects the campaign; `lib/grids.sh` documents all six. The default,
`pass2`, is the grid of arms this package pre-trains. `v1` is the same tasks
against released weights and is the reference column the rest is read against.

**`v1` needs weights this package does not produce.** Its arms read
`$JEPA_CKPT`, `$MUSICBERT_CKPT` and `$JEPA_CKPT_SEED2` — a second JEPA
pre-training seed, used as the diagnostic control that shows the collapsed
encoder belongs to `$JEPA_CKPT` and not to the adapter code. All three are
externally supplied and dropped into `$CHECKPOINT_ROOT` by you, and each can be
pointed elsewhere by exporting it. `./check_data.sh` reports which of the three
are absent. `pass2` has no such dependency for the four arms it trains:
`run_snapshot.sh` fills `$CHECKPOINT_ROOT` from your own training runs.

It does carry one *optional* fifth column, `musicbert` — the unchanged pass-1
reference, read from `$MUSICBERT_CKPT`, kept so that movement elsewhere in the
table can be read as movement in the retrained models rather than in the
harness. Without that file the job generator prints `SKIPPING arm musicbert`
and the report ends up saying *Arms not yet in this table: `musicbert`*. Both
lines are the intended behaviour on a machine that has no externally supplied
weights, not a failed run.

Two failure modes the design exists to prevent, both of which cost real time:

* **The encode array must read the RESOLVED config**, not a recipe. The job
  generator writes `_resolved_config.yaml` — checkpoint overrides already
  applied — next to the generated jobs, and the encode template reads only that.
  Pointed at a recipe instead, the cache reports thousands of hits in a minute,
  warms nothing, and the probe jobs re-encode the corpus inline.
* **The probe array depends on the encode array with `afterany`, not
  `afterok`.** A shard killed at the wall has still written every file it
  finished, and a probe falls back to encoding a miss inline; `afterok` would
  throw away a 95 %-warm cache over one bad shard.

### 6. Report

```bash
./run_report.sh                # the main grid
./run_report.sh composer
```

Prints the table and writes it as `CONSOLIDATED.md` next to the runs.
`report.format` is `json` in every recipe on purpose: the library's markdown
writer raises `NotImplementedError`, so the table is built from the JSON here.

---

## Smoke test

```bash
./run_smoke.sh                 # environments, submodule, and every recipe
./run_smoke.sh --with-data     # ...plus 2 training steps per arm, on CPU
./run_smoke.sh --end-to-end    # the whole pipeline, unattended, ~30 minutes
```

The recipe check is the one that catches the common mistake: a recipe names
`${SPLIT_DIR}/ids_train.txt`, and a shell that never sourced `env.sh` leaves it
unexpanded — which then fails as a missing file instead of silently training on
the wrong corpus.

**`--end-to-end` is the one to run before a campaign.** It cuts itself a
hundred-file pre-training corpus and a forty-movement CIPI corpus out of
whatever `$MIDI_DIR` and `$BENCHMIR_DATA_ROOT` point at, then runs *every stage
through the entry points above* — tokenize (both codecs) → union → pre-train all
four arms → snapshot → encode → probe → report — waiting for each SLURM stage
before starting the next, and prints what each cost:

```
     stage           sec
     corpus            1  ok
     tokenize         38  ok
     union             6  ok
     pre:jepa_paper  383  ok
     ...
```

Nothing in it is stubbed, so a pass means these commands work on *this* cluster,
in *these* environments, with *these* recipes. The table of accuracies it prints
at the end is meaningless by construction — thirty steps per arm, forty
movements, one unseeded probe fit — and is printed anyway, because a table that
renders is part of what is being proved.

It cannot disturb a real campaign: `WORK_ROOT` is redirected to
`$WORK_ROOT/smoke` and every derived path is rebuilt underneath it, so the smoke
gets its own caches, split, vocabulary, runs, checkpoints and jobs. In
particular it cannot repoint an `<arm>.ckpt` symlink that a running evaluation
grid is reading.

Knobs, should the defaults not suit your queue: `JR_SMOKE_FILES` (100),
`JR_SMOKE_SHARDS` (2), `JR_SMOKE_MOVEMENTS` (40), `JR_SMOKE_STEPS` (30),
`JR_SMOKE_TASK` (`cipi_difficulty`), `JR_SMOKE_POLL` (20 s). The usual
`JR_*_TIME` / `JR_*_MEM` knobs apply too; the smoke lowers each to something a
short queue will schedule.

---

## Expected outputs

Everything the package writes lives under `$WORK_ROOT`:

```
$WORK_ROOT/
├── cache/<codec>/                per-shard token caches
├── cache/union_<codec>_<tag>/    the union a training run reads (symlinks + index)
├── cache/embeddings/             per-bar embeddings, keyed by file content
│                                 plus a checkpoint salt
├── splits/<tag>/                 ids_{train,val,test}.txt, pieces_{train,val}.txt
├── vocab/                        dictionary_musescore.pkl
├── checkpoints/                  frozen snapshots: <arm>_step<N>.ckpt, <arm>.ckpt
├── runs/<arm>/                   training: checkpoints/, metrics.csv, summary.json
├── runs/<grid>/<job>/            evaluation: report.json per job
├── runs/<grid>/CONSOLIDATED.md   the results table
├── jobs/<grid>/                  generated job configs, _resolved_config.yaml,
│                                 _warm_tasks.txt, _checkpoints.txt
└── logs/                         one log per job
```

The embedding cache is keyed by **file content plus a checkpoint salt**, so an
intermediate checkpoint can never serve its embeddings to a later one, and two
arms can share the cache without colliding.

---

## Known limitations

Read these before quoting any number this package produces.

* **The probe is unseeded.** The evaluation library parses `run.seed` and never
  applies it to torch or numpy. Every cell is one unseeded fit, and every `±` is
  across folds, not across seeds. A difference smaller than the fold spread is
  not a difference.
* **Access to the evaluation library is its author's to grant.** If the
  submodule URL is not reachable for you, request access. The pre-training half
  of this package runs without it.
* **MuseTok cannot run without upstream's checkout**, which has no licence file
  and so is cloned by an opt-in flag rather than vendored. The other three arms
  are unaffected.
* **Our MuseTok checkpoints are not interchangeable with the public weights** —
  different vocabulary, shifted token ids.
* **A file the encoder refuses becomes a zero embedding row, not a dropped
  row.** It is a small fraction and it is not distributed evenly across arms, so
  a model that refuses more files is penalised twice.
* **Epoch budgets differ per arm by construction.** The JEPA arms are
  epoch-based; MuseTok and MusicBERT are step-based, so a smaller corpus raises
  their epoch counts. Rescaling MusicBERT means moving `trainer.max_steps`,
  `optim.total_num_update` **and** `optim.warmup_updates` together — move one
  and the learning-rate curve is discontinuous.
* **POP909 chord/root exist in two versions** — a 30-song historical slice and
  all 909 songs. They are ~18 k and ~550 k rows of the same probe. Never compare
  one with the other.
* **Corpus overlap between pre-training and evaluation is unmeasured.** The
  content-hash split removes leakage *inside* the pre-training corpus. It says
  nothing about a crowd-uploaded pre-training corpus containing arrangements of
  named evaluation pieces.
* **Compute.** The reference campaign used one GPU per arm for days, not hours,
  on a corpus of ~1.7 M files; evaluation is a few hundred CPU-hours. Pre-train
  with `--smoke` first, always.

---

## Licence

MIT — see `LICENSE`. Third-party components, corpora and their terms are in
`THIRD_PARTY_NOTICES.md`; none of them is redistributed here.
