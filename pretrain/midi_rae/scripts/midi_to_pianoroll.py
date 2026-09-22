#!/usr/bin/env python3
"""Convert filtered MIDI files into sharded binary piano-roll tensors for lazy-loading training.

Reads file_path column from the filtered dataset-stats CSV, loads the corresponding
.mid file from src-dir (MuseScore-big-MIDI layout: "<subfolder>/<id>.mxl" -> "<subfolder>/<id>.mxl.mid"),
rasterizes to a binary (128, T) piano roll at a fixed frame rate, and writes shards of
N records each (train_shard*.pt / val_shard*.pt) plus a sidecar *.idx.json with the record
count, so the lazy dataset (midi_rae.data.ShardedAnchorDataset / ShardedTripletDataset) can
build its index without loading full shard tensors.

Usage:
  python scripts/midi_to_pianoroll.py \
      --csv /home/abahuguna/soundofai/midi-rae/dataset_stats_filtered.csv \
      --src-dir /gpfs/scratch/mtg/abahuguna/midi-rae-jepa/raw_midi \
      --out-dir /gpfs/scratch/mtg/abahuguna/midi-rae-jepa/pianorolls \
      --fs 8 --max-len 2048 --shard-size 2000 --val-frac 0.02 --workers 32
"""
import argparse, os, json, random, csv as csvmod, warnings
import numpy as np
import torch
import pretty_midi
from multiprocessing import Pool

warnings.filterwarnings("ignore")


def midi_to_roll(path, steps_per_beat=8, max_len=4096, max_duration_sec=1800):
    """Tempo-normalized piano roll: columns are 1/steps_per_beat-note subdivisions
    (steps_per_beat=8 -> 32nd notes), matching the MIDI-RAE-JEPA papers' grid, rather
    than a fixed real-time frame rate. This keeps a given musical duration (in beats)
    at the same pixel width regardless of the piece's tempo."""
    pm = pretty_midi.PrettyMIDI(path)
    end_time = pm.get_end_time()
    if end_time <= 0 or end_time > max_duration_sec:
        return None
    tempi = pm.get_tempo_changes()[1]
    try:
        tempo = float(tempi[0]) if len(tempi) else pm.estimate_tempo()
    except Exception:
        tempo = 120.0
    if not (20 <= tempo <= 300):  # guard against bogus tempo estimates
        tempo = 120.0
    fs = (tempo / 60.0) * steps_per_beat  # columns per second, tempo-normalized
    roll = pm.get_piano_roll(fs=fs)  # (128, T) float velocities
    roll = (roll > 0).astype(np.uint8)
    if roll.shape[1] > max_len:
        roll = roll[:, :max_len]
    if roll.shape[1] < 8:  # too short to be useful
        return None
    return roll


def worker(args):
    rel_path, src_dir, steps_per_beat, max_len = args
    full_path = os.path.join(src_dir, rel_path + '.mid')
    try:
        return midi_to_roll(full_path, steps_per_beat=steps_per_beat, max_len=max_len)
    except Exception:
        return None


def write_shard(records, out_dir, split, shard_idx):
    path = os.path.join(out_dir, f'{split}_shard{shard_idx:05d}.pt')
    torch.save(records, path)
    with open(path.replace('.pt', '.idx.json'), 'w') as f:
        json.dump({'count': len(records)}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--src-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--steps-per-beat', type=int, default=8, help='subdivisions per beat (8 = 32nd notes, tempo-normalized)')
    ap.add_argument('--max-len', type=int, default=4096, help='max columns per piece (truncate longer pieces)')
    ap.add_argument('--shard-size', type=int, default=2000, help='records per shard file')
    ap.add_argument('--val-frac', type=float, default=0.02)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='debug: only process first N rows')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.csv, newline='') as f:
        reader = csvmod.DictReader(f)
        rel_paths = [row['file_path'] for row in reader]
    if args.limit:
        rel_paths = rel_paths[:args.limit]

    rng = random.Random(args.seed)
    shuffled = rel_paths[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * args.val_frac)
    val_paths, train_paths = shuffled[:n_val], shuffled[n_val:]

    stats = {'train_total': len(train_paths), 'val_total': len(val_paths),
             'train_ok': 0, 'val_ok': 0, 'failed': 0}

    for split, paths in [('val', val_paths), ('train', train_paths)]:
        buf = []
        shard_idx = 0
        job_args = [(p, args.src_dir, args.steps_per_beat, args.max_len) for p in paths]
        n_done = 0
        with Pool(args.workers) as pool:
            for roll in pool.imap(worker, job_args, chunksize=64):
                n_done += 1
                if roll is None:
                    stats['failed'] += 1
                else:
                    buf.append(roll)
                    stats[f'{split}_ok'] += 1
                    if len(buf) >= args.shard_size:
                        write_shard(buf, args.out_dir, split, shard_idx)
                        shard_idx += 1
                        buf = []
                if n_done % 20000 == 0:
                    print(f"[{split}] {n_done}/{len(paths)} processed, "
                          f"{stats[f'{split}_ok']} ok, {stats['failed']} failed so far", flush=True)
        if buf:
            write_shard(buf, args.out_dir, split, shard_idx)
            shard_idx += 1
        print(f"[{split}] wrote {shard_idx} shards, {stats[f'{split}_ok']} records", flush=True)

    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump({**stats, 'steps_per_beat': args.steps_per_beat, 'max_len': args.max_len}, f, indent=2)
    print("DONE", stats, flush=True)


if __name__ == '__main__':
    main()
