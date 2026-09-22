"""Populate the per-bar embedding cache for one model across every dataset.

`benchmir run` would do this itself, but its own embedding cache is written only
once a whole (dataset, model) extraction finishes -- so a job killed at the
walltime limit loses everything. This script drives the same encoders through
the same content-hash cache, one file at a time, so a killed job loses at most
one file and a resubmit picks up where it stopped.

Usage (pass 1, the four named checkpoints):
    warm_cache.py <music-jepa|musicbert|musetok|music-jepa-champA> [--device cuda]

Usage (pass 2 and anything later): name a config and a model_id and the
model is constructed from that config's own kwargs, so the warm run and the
eval run cannot disagree about which checkpoint, vocabulary or cache_name
is in play -- the failure mode this option exists to remove is a warm job
that fills one cache while the eval job reads another and silently
re-encodes (or worse, hits a stale entry):
    warm_cache.py --config configs/ours_all_evals_pass2.yaml \
                  --model-id jepa_musescore_paper-song --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The package's one path configuration (published_code/pkgpaths.py). It reads
# paths.yaml and the same environment variables env.sh exports, so a default
# printed by --help is the path this script will actually use.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pkgpaths import PATHS, expandvars_tree  # noqa: E402

RAW = PATHS.benchmir_data_root


def dataset_files(only: list[str] | None = None) -> list[Path]:
    """Every MIDI any of the six tasks will ask for, deduplicated.

    `only` restricts to named datasets. A single-task rerun (the official
    movement-level CIPI, say) otherwise pays for enumerating and encoding
    TopMAGD's 10,282 LMD files it will never look at.

    Each adapter is CONSTRUCTED LAZILY, behind a thunk, and only for the names
    that survive `only`. Building all six first and filtering afterwards made
    the flag useless in the case it was written for: every adapter reads its
    corpus's metadata in __init__, so warming cipi alone still raised
    FileNotFoundError for EMOPIA, POP909, TopMAGD and Humdrum. A site that holds
    one corpus, or a smoke that carries a cut-down copy of one, could not warm
    anything.
    """
    from benchmir.eval.datasets.corpora.cipi_eval_corpus import (
        CIPIDifficultyEstimationEvalDataset,
    )
    from benchmir.eval.datasets.corpora.emopia_eval_corpus import (
        EMOPIAEmotionClassificationEvalDataset,
    )
    from benchmir.eval.datasets.corpora.humdrum_eval_corpus import (
        HumdrumComposerClassificationEvalDataset,
    )
    from benchmir.eval.datasets.corpora.pop909cl_eval_corpus import (
        POP909ChordEstimationEvalDataset,
        POP909KeyEstimationEvalDataset,
    )
    from benchmir.eval.datasets.corpora.top_magd_eval_corpus import (
        MSDTopMAGDGenreClassificationEvalDataset,
    )

    datasets = [
        ("cipi", lambda: CIPIDifficultyEstimationEvalDataset(
            root_dir=RAW / "cipi", symbolic_format="midi", target_class="henle")),
        ("emopia", lambda: EMOPIAEmotionClassificationEvalDataset(
            root_dir=RAW / "emopia", target_class="4Q")),
        ("pop909-chord", lambda: POP909ChordEstimationEvalDataset(root_dir=RAW / "POP909cl")),
        ("pop909-key", lambda: POP909KeyEstimationEvalDataset(root_dir=RAW / "POP909cl")),
        ("topmagd", lambda: MSDTopMAGDGenreClassificationEvalDataset(
            root_dir=RAW / "LMDMatched", use_all_matches=False)),
        # Composer classification (configs/ours_composer_v1.yaml). Song level,
        # so the enumeration is the whole corpus and the fold does not matter:
        # every fold's train+val+test is the same 1,968 files, and the cache is
        # keyed by file content, not by split.
        #
        # max_examples_per_class is left at None ON PURPOSE. Upstream's debug
        # config caps it at 5; capping it here would warm ~100 of the 1,968
        # files and leave the probe jobs to encode the other ~1,870 inline.
        ("humdrum", lambda: HumdrumComposerClassificationEvalDataset(
            root_dir=RAW / "humdrum", symbolic_format="midi",
            target_class="composer")),
    ]
    if only:
        known = {n for n, _ in datasets}
        unknown = sorted(set(only) - known)
        if unknown:
            raise SystemExit(f"unknown dataset(s) {unknown}; have {sorted(known)}")
        datasets = [(n, f) for n, f in datasets if n in only]
    seen: dict[str, Path] = {}
    for name, factory in datasets:
        ds = factory()
        paths = [ds[i][0] for i in range(len(ds))]
        print(f"  {name}: {len(paths)} files", flush=True)
        for p in paths:
            seen.setdefault(str(p), Path(p))
    return list(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None,
                    choices=["music-jepa", "musicbert", "musetok", "music-jepa-champA"])
    ap.add_argument("--config", default=None,
                    help="a benchmir config; the model is built from its "
                         "`model:` entry for --model-id")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", action="append", default=[],
                    help="restrict to these dataset names (repeatable): cipi, "
                         "emopia, pop909-chord, pop909-key, topmagd, "
                         "humdrum")
    # Shard so the work can run as a CPU array when every GPU is taken. Files
    # are assigned round-robin, and the content-hash cache is shared, so shards
    # neither overlap nor need to be merged afterwards.
    ap.add_argument("--shard", default="0/1", help="i/N")
    args = ap.parse_args()

    from benchmir.models.ours_jepa import MusicJEPAEmbeddingModel
    from benchmir.models.ours_musetok import MuseTokEmbeddingModel
    from benchmir.models.ours_musicbert import MusicBERTEmbeddingModel

    if args.config:
        import yaml

        if not args.model_id:
            ap.error("--config requires --model-id")
        blob = expandvars_tree(yaml.safe_load(Path(args.config).read_text()))
        entry = next(
            (m for m in blob["model"] if m["model_id"] == args.model_id), None
        )
        if entry is None:
            ap.error(
                f"{args.model_id!r} is not in {args.config}; have "
                f"{[m['model_id'] for m in blob['model']]}"
            )
        by_class = {
            "MusicJEPAEmbeddingModel": MusicJEPAEmbeddingModel,
            "MuseTokEmbeddingModel": MuseTokEmbeddingModel,
            "MusicBERTEmbeddingModel": MusicBERTEmbeddingModel,
        }
        kwargs = dict(entry.get("kwargs") or {})
        kwargs["device"] = args.device      # the warm array is CPU-only
        kwargs["granularity"] = "song"      # granularity is not in the cache salt
        kwargs.pop("frames_per_bar", None)
        args.model = args.model_id
        model = by_class[entry["class"]](model_id=args.model_id, **kwargs)
        print(f"warming {args.model_id} from {args.config}: "
              f"cache_dir={model.cache_dir}", flush=True)
        return _run(args, model)

    if not args.model:
        ap.error("give a model name, or --config with --model-id")

    cls = {
        "music-jepa": MusicJEPAEmbeddingModel,
        "musicbert": MusicBERTEmbeddingModel,
        "musetok": MuseTokEmbeddingModel,
        "music-jepa-champA": MusicJEPAEmbeddingModel,
    }[args.model]
    kwargs = {}
    if args.model == "music-jepa-champA":
        # Second JEPA pretraining seed, kept as the control that proves the
        # collapse is in the default JEPA checkpoint, not in this code.  Like
        # $JEPA_CKPT it is supplied by you, not produced here; $JEPA_CKPT_SEED2
        # overrides the default location (see env.sh and check_data.sh).
        kwargs = {
            "checkpoint_path": os.environ.get(
                "JEPA_CKPT_SEED2",
                str(PATHS.checkpoint_root / "music_jepa_champA_s12.pt")),
            "cache_name": "music-jepa-champA",
        }
    model = cls(model_id=args.model, granularity="song", device=args.device, **kwargs)
    return _run(args, model)


def _run(args, model) -> None:
    print("collecting file list...", flush=True)
    files = dataset_files(args.dataset or None)
    if args.limit:
        files = files[: args.limit]
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    if shard_n > 1:
        files = files[shard_i::shard_n]
    print(
        f"{len(files)} unique MIDI files for {args.model} "
        f"(shard {shard_i}/{shard_n})",
        flush=True,
    )

    t0 = time.time()
    ok = failed = 0
    for i, path in enumerate(files, 1):
        blob = model._bars(path)
        if blob is None:
            failed += 1
        else:
            ok += 1
        if i % 200 == 0 or i == len(files):
            el = time.time() - t0
            rate = i / el if el else 0
            print(
                f"  {i}/{len(files)} ok={ok} failed={failed} "
                f"hits={model.client.n_hits} computed={model.client.n_computed} "
                f"{rate:.1f} files/s eta={(len(files)-i)/rate/60:.1f}min",
                flush=True,
            )

    out = (PATHS.log_root
           / f"warm_{args.model}_s{shard_i}of{shard_n}_failures.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model.client.failures, indent=1))
    print(f"done: ok={ok} failed={failed} ({out})", flush=True)
    model.client.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
