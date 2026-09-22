"""Split geometry + baselines for TopMAGD under each split_strategy.

Reports, per (use_all_matches, split_strategy):
  - indexed rows and unique songs
  - train/val/test sizes per fold
  - the two baselines the probe has to beat on TEST: majority-class accuracy
    and 1/num_classes chance.

Baselines are computed on the split that is actually scored (TEST), not on the
whole corpus: balanced_fixed_size moves 2,000 tracks/genre into TRAIN, which
leaves a TEST set with a *different* skew from the corpus, so a corpus-wide
majority number would flatter or punish the wrong column.

Usage:  split_stats.py [out.json]

With no argument it writes $WORK_ROOT/topmagd_split_stats.json, which is what
make_report_topmagd_splits.py reads. ./run_report.sh topmagd runs this first.
"""
import collections
import json
import sys
from pathlib import Path

# The package's one path configuration (published_code/pkgpaths.py). It reads
# paths.yaml and the same environment variables env.sh exports, so a default
# printed here is the path this script will actually use.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pkgpaths import PATHS  # noqa: E402

sys.path.insert(0, str(PATHS.benchmir_root / "src"))
from benchmir.eval.datasets.corpora.top_magd_eval_corpus import (  # noqa: E402
    MSDTopMAGDGenreClassificationEvalDataset as DS,
)

ROOT = str(PATHS.benchmir_data_root / "LMDMatched") + "/"
out = {}

for uam in (False, True):
    for strat in ("stratified_kfold", "balanced_fixed_size", "stratified_percentage_split"):
        kw = dict(root_dir=ROOT, use_all_matches=uam, split_strategy=strat, val_ratio=0.15)
        if strat == "stratified_kfold":
            kw["n_folds"] = 5
        ds = DS(**kw)
        labels = [g for _, g in ds._index]
        songs = {p.parent.name for p, _ in ds._index}
        key = f"use_all_matches={uam}|{strat}"
        folds = {}
        for fid in ds.get_fold_ids():
            v = ds.fold_view(fid)
            sp = v.get_splits()
            test_labels = [labels[i] for i in sp["test"]]
            c = collections.Counter(test_labels)
            maj_name, maj_n = c.most_common(1)[0]
            folds[fid] = dict(
                train=len(sp["train"]), val=len(sp["val"]), test=len(sp["test"]),
                test_classes=len(c),
                majority_class=maj_name,
                majority_acc=maj_n / len(test_labels),
                chance=1.0 / ds.num_classes,
                train_class_counts=dict(collections.Counter(labels[i] for i in sp["train"])),
                test_class_counts=dict(c),
            )
        covered = sum(f["train"] + f["val"] + f["test"] for f in folds.values())
        out[key] = dict(
            indexed_rows=len(ds._index), unique_songs=len(songs),
            num_classes=ds.num_classes, class_labels=ds.class_labels,
            corpus_class_counts=dict(collections.Counter(labels)),
            folds=folds,
            rows_used_fold0=folds[ds.get_fold_ids()[0]]["train"]
                            + folds[ds.get_fold_ids()[0]]["val"]
                            + folds[ds.get_fold_ids()[0]]["test"],
        )
        print(f"{key}: rows={len(ds._index)} songs={len(songs)} classes={ds.num_classes} "
              f"folds={list(folds)} fold0={folds[ds.get_fold_ids()[0]]['train']}/"
              f"{folds[ds.get_fold_ids()[0]]['val']}/{folds[ds.get_fold_ids()[0]]['test']} "
              f"maj={folds[ds.get_fold_ids()[0]]['majority_acc']:.4f}", flush=True)

# Default to the single location make_report_topmagd_splits.py reads, so
# neither script depends on the operator knowing the filename.
dest = Path(sys.argv[1]) if len(sys.argv) > 1 else PATHS.work_root / "topmagd_split_stats.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1))
print("wrote", dest)
