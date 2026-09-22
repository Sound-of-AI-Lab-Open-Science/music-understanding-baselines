"""One place that knows, per baseline: which cache it eats, which
LightningModule/DataModule pair to build, and which Lightning package to use.

WHY the Lightning package matters: Music-JEPA's DataModule subclasses
``pytorch_lightning``'s base class while MusicBERT's subclasses ``lightning``'s.
The two are separate class trees in this env, and mixing a Trainer from one with
a DataModule from the other makes Lightning raise "Expected a parent".  Each
builder therefore reports the package its objects came from and ``train.py``
instantiates the matching Trainer.

All imports are local to the builders: Music-JEPA and MusicBERT run in the
``jepa-music`` env, MuseTok in the ``musetok`` env, and neither env can import
the other's dependencies.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

MODELS = ("jepa", "musicbert", "musetok", "midi_rae_enc", "midi_rae_dec")

#: Which tokenization each model consumes.  jepa and musicbert deliberately share
#: one Octuple cache directory, so a corpus is tokenized once for both.  The two
#: midi_rae arms share one piano-roll cache the same way.
CACHE_KIND = {"jepa": "octuple", "musicbert": "octuple", "musetok": "remi",
              "midi_rae_enc": "pianoroll", "midi_rae_dec": "pianoroll"}

#: Conda interpreter that can run each model, as an unexpanded template: the
#: names come from ``paths.yaml`` and the SLURM templates resolve them through
#: ``env.sh``.  Documentation only -- nothing here executes it.
ENV_PYTHON = {
    "jepa": "${CONDA_ROOT}/envs/${JEPA_ENV}/bin/python",
    "musicbert": "${CONDA_ROOT}/envs/${MUSICBERT_ENV}/bin/python",
    "musetok": "${CONDA_ROOT}/envs/${MUSETOK_ENV}/bin/python",
    "midi_rae_enc": "${CONDA_ROOT}/envs/${MIDI_RAE_ENV}/bin/python",
    "midi_rae_dec": "${CONDA_ROOT}/envs/${MIDI_RAE_ENV}/bin/python",
}


@dataclass
class BuiltRun:
    """Everything ``train.py`` needs to run one model, package-consistent."""

    lightning: Any            # the ``lightning``/``pytorch_lightning`` module itself
    callbacks_mod: Any        # its ``.callbacks`` namespace
    loggers_mod: Any          # its ``.loggers`` namespace
    datamodule: Any
    module: Any
    monitor: Optional[str]    # metric for the best-checkpoint callback, or None
    step_metrics: Sequence[str]   # what the ASCII step logger should print


# ---------------------------------------------------------------------------
# tokenization caches
# ---------------------------------------------------------------------------
def cache_dir_for(model: str, cache_root: str, files: Sequence[str]) -> str:
    """Deterministic cache location for ``model`` over exactly ``files``.

    Branches on ``kind`` BEFORE importing anything, so a ``pianoroll`` model
    (midi_rae) never touches ``baselines.data.octuple_cache`` -- which does not
    exist in this checkout (see registry's module docstring situation: jepa /
    musicbert / musetok's own cache modules are out of scope for this change
    and are left exactly as missing as they were found).
    """
    kind = CACHE_KIND[model]
    if kind == "pianoroll":
        from baselines.data.pianoroll_cache import fingerprint
    else:
        from baselines.data.octuple_cache import fingerprint
    return os.path.join(cache_root, kind, fingerprint(files, extra=kind))


def prepare_cache(model: str, files: Sequence[str], midi_dir: str, out_dir: str, *,
                  workers: int = 8, force: bool = False,
                  config: Optional[dict] = None,
                  resolve_vocab: bool = True) -> Dict:
    """Tokenize ``files`` into ``out_dir`` with the codec ``model`` needs.

    ``resolve_vocab=False`` stops short of attaching the MuseTok dictionary.
    Tokenizing writes REMI+ event pickles and nothing else -- no model is built
    and no token id is looked up -- but the corpus dictionary is DERIVED FROM
    those pickles, by ``run_union.sh`` step 3, over every shard at once. So
    demanding it here makes the package's own documented order
    (``run_tokenize.sh remi`` then ``run_union.sh``) impossible to run on a
    fresh tree: the first REMI+ shard dies with FileNotFoundError on the very
    file the step after it exists to write. ``train.py --tokenize-only`` passes
    False; every path that goes on to build a model leaves it True, so a real
    run still fails loudly and early on a missing dictionary.
    """
    kind = CACHE_KIND[model]
    if kind == "pianoroll":
        from baselines.data.pianoroll_cache import build_pianoroll_cache

        data_cfg = (config or {}).setdefault("data", {})
        return build_pianoroll_cache(
            files, out_dir, workers=workers, force=force,
            steps_per_beat=int(data_cfg.pop("steps_per_beat", 8)),
            max_len=int(data_cfg.pop("max_len", 4096)),
            shard_size=int(data_cfg.pop("shard_size", 2000)),
            val_frac=float(data_cfg.pop("val_frac", 0.02)),
            seed=int(data_cfg.pop("cache_seed", 42)))

    if kind == "octuple":
        from baselines.data.octuple_cache import build_octuple_cache

        # popped, not read: data is splatted into DataModule constructors that
        # reject unknown keys.
        dedup = bool((config or {}).setdefault("data", {}).pop("dedup", False))
        return build_octuple_cache(files, out_dir, workers=workers, dedup=dedup,
                                   force=force)

    from baselines.data.remi_cache import build_remi_cache

    info = build_remi_cache(files, midi_dir, out_dir, workers=workers, force=force)
    if not resolve_vocab:
        return info
    return _resolve_remi_vocab(info, out_dir, config, force=force)


def _resolve_remi_vocab(info: Dict, out_dir: str, config: Optional[dict], *,
                        force: bool = False) -> Dict:
    """Attach ``vocab_path`` / ``vocab_size`` to a REMI cache dict.

    Shared by the tokenize path and the ``--cache-prebuilt`` path so a union
    cache resolves its vocabulary exactly like a freshly built one.
    """
    from baselines.data.remi_cache import DEFAULT_VOCAB_PATH, build_vocab, vocab_size

    data_cfg = (config or {}).setdefault("data", {})
    vocab_path = data_cfg.pop("vocab_path", None) or DEFAULT_VOCAB_PATH
    if data_cfg.pop("rebuild_vocab", False):
        vocab_path = os.path.join(out_dir, "dictionary.pkl")
        if force or not os.path.isfile(vocab_path):
            build_vocab(info["events_dir"], vocab_path)
    info["vocab_path"] = vocab_path
    info["vocab_size"] = vocab_size(vocab_path)
    print("[remi] vocab {} -> n_token {}".format(vocab_path, info["vocab_size"]),
          flush=True)
    return info


def load_prebuilt_cache(model: str, cache_dir: str,
                        config: Optional[dict] = None) -> Dict:
    """Load a union cache built by ``baselines/cache_tools/merge_shard_caches.py``.

    Returns the same dict shape ``prepare_cache`` produces, so ``build`` cannot
    tell a merged corpus from a freshly tokenized one.  No MIDI is read.
    """
    import json

    kind = CACHE_KIND[model]
    index_path = os.path.join(cache_dir, "index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError("no index.json under {}".format(cache_dir))
    with open(index_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    meta = blob["meta"]
    got = meta.get("kind", kind)
    if got != kind:
        raise ValueError("cache at {} is {!r} but {} needs {!r}".format(
            cache_dir, got, model, kind))
    if kind == "pianoroll":
        # No per-kind import needed: the piano-roll cache's meta dict (written
        # by build_pianoroll_cache) already has everything the sharded
        # datamodule needs (shard_dir == cache_dir itself).
        print("[pianoroll] prebuilt cache: {} sequences at {}".format(
            meta.get("num_sequences"), cache_dir), flush=True)
        return {"cached": True, "out_dir": os.path.abspath(cache_dir), **meta}
    if kind == "octuple":
        # Same pop prepare_cache() does: `dedup` is a tokenization-time option,
        # and JepaDataConfig/the MLM DataModule reject any key they do not
        # declare -- leaving it in `data` would crash the prebuilt path only.
        (config or {}).setdefault("data", {}).pop("dedup", None)
        print("[octuple] prebuilt cache: {} sequences at {}".format(
            meta.get("num_sequences"), cache_dir), flush=True)
        return {"cached": True, "out_dir": os.path.abspath(cache_dir), **meta}

    info = {"cached": True,
            "events_dir": os.path.join(os.path.abspath(cache_dir), "events"),
            "pieces": blob["pieces"], **meta}
    print("[remi] prebuilt cache: {} pieces at {}".format(
        meta.get("num_pieces"), info["events_dir"]), flush=True)
    return _resolve_remi_vocab(info, os.path.abspath(cache_dir), config)


# ---------------------------------------------------------------------------
# model builders
# ---------------------------------------------------------------------------
def _apply_octuple_id_split(dm, train_ids_file: str, val_ids_file: str) -> None:
    """Replace the seeded val split with an explicit SONG-LEVEL id split.

    Done entirely on the baselines side: ``JepaDataConfig`` is a dataclass that
    rejects unknown keys, so the two file paths are popped from the YAML before
    ``upstream.build`` sees them, and the split is applied afterwards by
    re-indexing the already-constructed datasets.  Nothing in ``src/`` changes.

    The min_notes filter and the ``data.fraction`` ablation still apply: we only
    re-partition the indices upstream already deemed valid.
    """
    from baselines.cache_tools.merge_shard_caches import id_of, load_keep_ids

    dm.setup()   # idempotent -- Lightning's later setup() call returns early
    train_ids = load_keep_ids(train_ids_file)
    val_ids = load_keep_ids(val_ids_file)
    seqs = dm.base.sequences
    valid = sorted(set(dm.train_set.indices) | set(dm.val_set.indices))
    train_idx, val_idx, unassigned = [], [], 0
    for i in valid:
        sid = id_of(seqs[i].get("name") or seqs[i].get("path") or seqs[i]["md5"])
        if sid in train_ids:
            train_idx.append(i)
        elif sid in val_ids:
            val_idx.append(i)
        else:
            unassigned += 1
    if not train_idx:
        raise ValueError("explicit id split left 0 training sequences: do the ids "
                         "in {} match the cache? (expected ids like '100000')"
                         .format(train_ids_file))
    dm.train_set.indices = train_idx
    dm.val_set.indices = val_idx
    print("[jepa-data] explicit id split: {} train / {} val sequences "
          "({} cached sequences in neither list, dropped)".format(
              len(train_idx), len(val_idx), unassigned), flush=True)


def _build_jepa(config: dict, cache: Dict) -> BuiltRun:
    import pytorch_lightning as pl
    from pytorch_lightning import callbacks, loggers

    import scripts.train_jepa as upstream   # reuse the repo's own config -> objects

    data_cfg = config.setdefault("data", {})
    # Popped BEFORE upstream.build: JepaDataConfig is a dataclass and raises on
    # any key it does not declare.
    train_ids_file = data_cfg.pop("train_ids_file", None)
    val_ids_file = data_cfg.pop("val_ids_file", None)
    if bool(train_ids_file) != bool(val_ids_file):
        raise ValueError("data.train_ids_file and data.val_ids_file must be given "
                         "together (or neither)")
    data_cfg["data_root"] = cache["out_dir"]
    dm, lit = upstream.build(config)
    if train_ids_file:
        _apply_octuple_id_split(dm, train_ids_file, val_ids_file)
    return BuiltRun(pl, callbacks, loggers, dm, lit, "val/loss",
                    ("train/loss_step", "train/loss_cos_step", "train/perdim_var_step"))


def _apply_mlm_id_split(dm, train_ids_file: str, val_ids_file: str,
                        val_max_samples: int) -> None:
    """Same song-level id split as :func:`_apply_octuple_id_split`, for the
    MusicBERT MLM datamodule (``MlmDataset(_Subset(base, indices))``)."""
    from baselines.cache_tools.merge_shard_caches import id_of, load_keep_ids

    dm.setup()
    train_ids = load_keep_ids(train_ids_file)
    val_ids = load_keep_ids(val_ids_file)
    sub_tr, sub_va = dm.train_set.base, dm.val_set.base
    base = sub_tr.base
    seqs = base.sequences
    valid = sorted(set(sub_tr.indices) | set(sub_va.indices))
    train_idx, val_idx, unassigned = [], [], 0
    for i in valid:
        sid = id_of(seqs[i].get("name") or seqs[i].get("path") or seqs[i]["md5"])
        if sid in train_ids:
            train_idx.append(i)
        elif sid in val_ids:
            val_idx.append(i)
        else:
            unassigned += 1
    if not train_idx:
        raise ValueError("explicit id split left 0 training sequences for MusicBERT: "
                         "do the ids in {} match the cache?".format(train_ids_file))
    if val_max_samples and len(val_idx) > val_max_samples:
        val_idx = val_idx[:val_max_samples]   # validation is a monitor, keep it cheap
    sub_tr.indices = train_idx
    sub_va.indices = val_idx
    print("[musicbert-data] explicit id split: {} train / {} val sequences "
          "({} in neither list, dropped)".format(len(train_idx), len(val_idx), unassigned),
          flush=True)


def _build_musicbert(config: dict, cache: Dict) -> BuiltRun:
    import lightning as L
    from lightning.pytorch import callbacks, loggers

    import scripts.train_mlm as upstream

    data_cfg = config.setdefault("data", {})
    data_cfg["data_dir"] = cache["out_dir"]
    train_ids_file = data_cfg.pop("train_ids_file", None)
    val_ids_file = data_cfg.pop("val_ids_file", None)
    if bool(train_ids_file) != bool(val_ids_file):
        raise ValueError("data.train_ids_file and data.val_ids_file must be given together")
    dm, lit = upstream.build(config)
    if train_ids_file:
        _apply_mlm_id_split(dm, train_ids_file, val_ids_file,
                            int(data_cfg.get("val_max_samples", 512)))
    return BuiltRun(L, callbacks, loggers, dm, lit, "val/loss",
                    ("train/loss_step", "train/acc_step"))


def _load_piece_list(path: Optional[str]) -> Optional[list]:
    """Piece names from a JSON list or a newline-delimited file (or None)."""
    if not path:
        return None
    import json

    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            blob = json.load(f)
            items = blob if isinstance(blob, list) else blob.get("pieces", [])
            return [str(x) for x in items]
        return [ln.strip() for ln in f if ln.strip()]


def _build_musetok(config: dict, cache: Dict) -> BuiltRun:
    import lightning as L
    from lightning.pytorch import callbacks, loggers

    from baselines.data.musetok_datamodule import MuseTokDataModule
    from baselines.models.musetok_module import MuseTokLitModule

    data_cfg = dict(config.get("data", {}))
    # Explicit train/val piece lists (names as they appear in the merged index,
    # e.g. "s03/100000.mxl.pkl").  Anything the cache does not actually hold is
    # dropped here rather than crashing REMIEventDataset on a missing pickle.
    train_pieces = _load_piece_list(data_cfg.get("train_pieces_file"))
    val_pieces = _load_piece_list(data_cfg.get("val_pieces_file"))
    if train_pieces is not None and val_pieces is not None:
        have = set(cache["pieces"])
        kept_tr = [p for p in train_pieces if p in have]
        kept_va = [p for p in val_pieces if p in have]
        print("[musetok] explicit split: train {}/{} val {}/{} pieces present in "
              "the cache".format(len(kept_tr), len(train_pieces),
                                 len(kept_va), len(val_pieces)), flush=True)
        train_pieces, val_pieces = kept_tr, kept_va
    elif train_pieces is not None or val_pieces is not None:
        raise ValueError("data.train_pieces_file and data.val_pieces_file must be "
                         "given together (or neither)")

    dm = MuseTokDataModule(
        cache["events_dir"], cache["vocab_path"], cache["pieces"],
        batch_size=int(data_cfg.get("batch_size", 24)),
        max_bars=int(data_cfg.get("max_bars", 16)),
        enc_seqlen=int(data_cfg.get("enc_seqlen", 128)),
        dec_seqlen=int(data_cfg.get("dec_seqlen", 1280)),
        do_augment=bool(data_cfg.get("do_augment", True)),
        val_fraction=float(data_cfg.get("val_fraction", 0.02)),
        val_max_pieces=int(data_cfg.get("val_max_pieces", 256)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        seed=int(config.get("seed", 1)),
        balanced_density=bool(data_cfg.get("balanced_density", False)),
        density_path=data_cfg.get("density_path"),
        train_pieces=train_pieces, val_pieces=val_pieces,
    )
    dm.setup()
    optim_cfg = dict(config.get("optim", {}))
    lit = MuseTokLitModule(
        vocab_size=dm.vocab_size,
        model=config.get("model", {}),
        beta=float(optim_cfg.get("beta", 1.0)),
        max_lr=float(optim_cfg.get("max_lr", 1e-4)),
        min_lr=float(optim_cfg.get("min_lr", 5e-6)),
        warmup_steps=int(optim_cfg.get("lr_warmup_steps", 200)),
        decay_steps=int(optim_cfg.get("lr_decay_steps", 150000)),
    )
    monitor = "val/loss" if dm.val_set else None
    return BuiltRun(L, callbacks, loggers, dm, lit, monitor,
                    ("train/loss_step", "train/recons_loss_step",
                     "train/commit_loss_step", "train/acc_step"))


def _build_midi_rae_enc(config: dict, cache: Dict) -> BuiltRun:
    import lightning as L
    from lightning.pytorch import callbacks, loggers

    from baselines.data.pianoroll_datamodule import PianorollTripletDataModule
    from baselines.models.midi_rae_encoder_module import MidiRaeEncoderModule

    data_cfg = dict(config.get("data", {}))
    training_cfg = dict(config.get("training", {}))
    dm = PianorollTripletDataModule(
        cache["out_dir"],
        batch_size=int(training_cfg.get("batch_size", 300)),
        max_shift_x=int(training_cfg.get("max_shift_x", 12)),
        max_shift_y=int(training_cfg.get("max_shift_y", 12)),
        num_workers=training_cfg.get("num_workers", (6, 2)))
    lit = MidiRaeEncoderModule(model=config.get("model", {}), training=training_cfg,
                               image_size=int(data_cfg.get("image_size", 128)))
    return BuiltRun(L, callbacks, loggers, dm, lit, "val/loss",
                    ("train/loss_step", "train/sim_step", "train/sigreg_step",
                     "train/mep_step"))


def _build_midi_rae_dec(config: dict, cache: Dict) -> BuiltRun:
    import lightning as L
    from lightning.pytorch import callbacks, loggers

    from baselines.data.pianoroll_datamodule import PianorollAnchorDataModule
    from baselines.models.midi_rae_decoder_module import MidiRaeDecoderModule

    data_cfg = dict(config.get("data", {}))
    training_cfg = dict(config.get("training", {}))
    encoder_ckpt = config.get("encoder_ckpt")
    if not encoder_ckpt:
        raise ValueError("midi_rae_dec requires top-level config key 'encoder_ckpt' "
                         "(the Lightning checkpoint written by a midi_rae_enc run)")
    dm = PianorollAnchorDataModule(
        cache["out_dir"],
        batch_size=int(training_cfg.get("dec_batch_size", 360)),
        max_shift_y=int(training_cfg.get("max_shift_y", 12)),
        num_workers=training_cfg.get("num_workers", (6, 4)))
    lit = MidiRaeDecoderModule(model=config.get("model", {}), training=training_cfg,
                               image_size=int(data_cfg.get("image_size", 128)),
                               encoder_ckpt=encoder_ckpt)
    return BuiltRun(L, callbacks, loggers, dm, lit, "val/loss",
                    ("train/loss_step", "train/bce_step", "train/mse_step"))


_BUILDERS = {"jepa": _build_jepa, "musicbert": _build_musicbert,
             "musetok": _build_musetok,
             "midi_rae_enc": _build_midi_rae_enc, "midi_rae_dec": _build_midi_rae_dec}


def build(model: str, config: dict, cache: Dict) -> BuiltRun:
    """Construct the (DataModule, LightningModule) pair for ``model``."""
    if model not in _BUILDERS:
        raise ValueError("unknown model {!r}; choose from {}".format(model, MODELS))
    return _BUILDERS[model](config, cache)
