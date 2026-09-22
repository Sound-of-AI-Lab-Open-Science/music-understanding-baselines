# The pre-training scaffold

Three pre-training baselines behind one CLI, so that adding a corpus is a path
and not a port:

| model | objective | tokenization | environment |
|---|---|---|---|
| `jepa` | Music-JEPA joint-embedding prediction | OctupleMIDI | `$JEPA_ENV` |
| `musicbert` | MusicBERT masked LM | OctupleMIDI (**the same cache as `jepa`**) | `$JEPA_ENV` |
| `musetok` | MuseTok residual-VQ reconstruction | REMI+ events | `$MUSETOK_ENV` |

```
python baselines/train.py --model {jepa,musicbert,musetok} \
    --config baselines/configs/<arm>.yaml \
    --midi-dir <dir> [--filter-csv <csv>] [--max-steps N] [--smoke]
```

Run it from `pretrain/`, which must be on `PYTHONPATH` — `env.sh` puts it
there — because `train.py` resolves both `baselines.*` and the `src.*` /
`scripts.*` packages next to it. You normally do not run it by hand:
`../run_pretrain.sh <arm>` picks the config, the interpreter and the union cache
for an arm and submits it.

`baselines/` is the scaffold; the model code it builds through lives in
`src/` (the architectures, the Octuple codec, VICReg, EMA) and in
`pretrain/scripts/train_jepa.py`, `pretrain/scripts/train_mlm.py` and
`pretrain/scripts/preprocess_lmd.py` (config → objects, and the parity-tested Octuple
encoder). The scaffold calls those rather than re-implementing them, so a
recipe here and a hand-written config produce the same objects.

**Never create a directory under `baselines/` whose name collides with a
top-level package next to it** (`scripts`, `src`, `data`, `models`, `configs`).
Such a directory would shadow the sibling package and break imports. That is
why the cache tooling lives in `baselines/cache_tools/`.

A run does, in order: discover MIDI → apply the filter CSV → build (or reuse)
the on-disk tokenization cache → build the model → fit → write a checkpoint.
Tokenization is cached per (codec, exact file list), so the second run of any
model over the same corpus starts training immediately.

Every string in a config is passed through `os.path.expandvars`, so a recipe can
write `${SPLIT_DIR}/ids_train.txt` and stay relocatable. An undefined variable
is left verbatim and fails as a missing file — loudly, rather than silently
falling back to a different corpus.

## Which interpreter

The three environments are not interchangeable. `$MUSETOK_ENV` pins
`miditoolkit==1.0.0` (1.0.1 changes tick handling and breaks the strict REMI+
conversion) and `vector_quantize_pytorch`; `$JEPA_ENV` has neither and has this
package's own `src/` dependencies. `env.sh` resolves both names to interpreters.

## Expected data layout

* `--midi-dir` — a tree of `.mid` / `.midi` files, searched **recursively**;
  nesting and duplicate basenames across subdirectories are fine.
* `--filter-csv` — optional. A CSV with a header row that restricts the corpus.
  Its column names differ per corpus, so they are configurable per model under
  `data.filter`:

  ```yaml
  data:
    filter:
      id_column: null        # null -> the CSV's FIRST column
      match: stem            # id | stem | basename | relpath | relpath_stem
      keep_columns: []       # each must be truthy to keep the file
      drop_columns: []       # any one truthy drops the file
      require_listed: true   # a file absent from the CSV is dropped
  ```

  `match` says how a MIDI path is reduced to the key compared against the id
  column. For the file `<midi-dir>/0/100000.mxl.mid`:

  | mode | key | use for |
  |---|---|---|
  | `id` | `100000` | corpora whose files are `<id>.<something>.mid`, where `stem` would leave `100000.mxl` |
  | `stem` (default) | `100000.mxl` | corpora with a single `.mid` extension, where stem == id |
  | `basename` | `100000.mxl.mid` | |
  | `relpath` | `0/100000.mxl.mid` | ids that carry their subdirectory |
  | `relpath_stem` | `0/100000.mxl` | |

  The CSV id is reduced the same way, so a `path` column holding a nested
  filename still matches a `stem`-matched file.

  Values `""`, `0`, `false`, `no`, `nan`, `none` count as false
  (case-insensitive); anything else counts as true, including the literal
  strings `True` / `False`. A run prints the CSV's real column names, its
  **data-row count and its lookup-key count separately** (one row is indexed
  under several keys, so the two differ), the distinct values in every keep/drop
  column, and how many files matched or were dropped and why. `--filter-only`
  stops right there, so a filter can be validated against a corpus of millions
  of files without tokenizing it.

  **Do not set `data.filter` in a recipe for a corpus restriction.** The filter
  feeds the cache fingerprint, so it forces a full re-tokenization. Restrict
  afterwards, with a keep-list on the union cache — see below.

## Tokenization cache

Tokenization runs **once** per (codec, exact file list) and is reused:

```
<cache-root>/octuple/<N>-<hash>/   index.json + tokens_NNNNN.npy + failures.txt
<cache-root>/remi/<N>-<hash>/      events/<piece>.pkl + index.json + failures.txt
```

The hash covers the filtered file list, so changing the filter builds a new
cache and leaves the old one intact. `jepa` and `musicbert` land in the *same*
Octuple directory — tokenize once, train both.

Per-file failures are logged and skipped, never fatal: a run prints a
`skip reasons` histogram and writes one `reason<TAB>path` line per dropped file
to `failures.txt`. `--force-tokenize` rebuilds; `--tokenize-only` stops after the
cache is built.

### Union cache: many shard caches, one corpus

A large corpus is tokenized as one array task per shard, which leaves one cache
directory per shard per codec. `cache_tools/merge_shard_caches.py` merges them
into one trainable corpus **without re-tokenizing** (`../run_union.sh` drives
it). The merge is index-only: Octuple `tokens_NNNNN.npy` become relative
symlinks, renumbered globally with each sequence's `shard` field rewritten;
REMI+ gets ONE symlink per source shard (`s00`, `s01`, …) and piece names become
`s<NN>/<piece>.pkl`. A keep-list therefore costs seconds instead of the days a
re-tokenization would. A shard directory whose `index.json` is missing or empty
— a job still running — is skipped loudly, never merged silently.

`--cache-prebuilt` skips discovery, filtering and tokenization entirely; it does
not even walk `--midi-dir`, which is then not required. It refuses to be
combined with `--filter-csv`, `--tokenize-only` or `--filter-only`.

**Duplicate ids (`--dedup-ids`, default `prefer-mxl`).** Such corpora are not
one file per id: a large fraction of ids carry two byte-identical files under
different extensions in the same shard, so that fraction of the corpus is
tokenized twice. The merge collapses each id to one item, keeping the
`.mxl`-derived name. `--dedup-ids none` keeps every item — Octuple still drops
exact md5 collisions, because the md5 in an Octuple index hashes the FILE BYTES,
so byte-identical twins collide there anyway. Both drop counters are printed on
every run.

**Song-level split (`--split-dir`).** Point it at a
`pretrain/scripts/make_musescore_split.py` output directory. The union is restricted to
train+val ids — **test ids never enter the cache**, so a held-out song is not
even reachable from the training corpus — and for REMI+ the merge writes
`pieces_train.txt` / `pieces_val.txt` back into that directory, holding the
MERGED piece names of the ids that survived tokenization. Only the merge knows
the `s<NN>` prefix, which is why it is the thing that emits them.
`ids_test.txt` is read, never rewritten.

Explicit splits at train time:

* **MuseTok** — `data.train_pieces_file` / `data.val_pieces_file` (the two files
  the merge wrote) replace the seeded shuffle; `data.val_max_pieces` still caps
  the validation list and the log says so when it truncates.
* **JEPA** — `data.train_ids_file` / `data.val_ids_file` re-partition the
  Octuple sequences by id after the DataModule is built, so `min_notes` and the
  `data.fraction` ablation still apply. Nothing in `src/` changes.
* **MusicBERT** — its DataModule owns its own split.

Without those keys every model falls back to the seeded `val_fraction` shuffle,
which mixes the split's validation songs into train.

## The recipes

`configs/*.yaml` carry the published hyper-parameters as defaults and say in
their header comment where each number came from and which deviations are
deliberate. Each file also has a `smoke:` section that is deep-merged over the
rest when `--smoke` is passed: production batch size, sequence length and
precision are kept, so a smoke's measured throughput extrapolates; only the step
count and the split wiring change.

## Outputs

```
<out-dir>/
    checkpoints/{last,final}.ckpt, <model>-best-*.ckpt, <model>-periodic-*.ckpt
    version_0/metrics.csv        per-step CSV log
    summary.json                 steps, wall time, final metrics, cache path
```

A periodic checkpoint is written unconditionally (`monitor=None`), so a run
killed at the walltime is resumable even if validation never ran. The training
templates pass `--resume`, which picks up `checkpoints/last.ckpt`.

## Known limits and assumptions

* **Filter-CSV column names must be chosen per corpus.** Validate any new CSV
  with `--filter-only` before committing to a long run.
* **MuseTok `balanced_density` sampling is OFF.** Upstream oversamples by note
  density using a pickle built over the specific corpus, which cannot exist
  before the dataset does. Build it, then set `data.balanced_density: true` and
  `data.density_path` — and RESTART, because it changes sampling.
* **MuseTok's vocabulary is corpus-derived here**, not upstream's released
  168-token dictionary, because roughly a quarter of a broad corpus carries
  events that dictionary cannot express. Token ids shift: checkpoints trained
  here are NOT interchangeable with the released MuseTok weights.
* **Upstream's `training.device` is dropped**; Lightning owns device placement
  and a job is given exactly one GPU.
* **Do not run MuseTok outside a job.** 79 M parameters with a 1280-token
  decoder is past what a shared login node tolerates; a CPU-only two-step
  attempt there gets killed. On a small GPU the same work takes seconds.
* **This scaffold only pre-trains.** Evaluation lives under `eval/`.
* **Upstream MuseTok code is imported, not vendored** — see
  `THIRD_PARTY_NOTICES.md`. Moving or editing that checkout changes behaviour
  here.
