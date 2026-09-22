"""Sharded, lazily-loaded piano-roll datasets, ported from upstream
``midi_rae/data.py`` (see package __init__). This is the HAND-WRITTEN part of
upstream's ``data.py`` (not nbdev-exported) -- ``ShardedAnchorDataset`` /
``ShardedTripletDataset`` plus their shared helpers -- written to support
training on the full MuseScore-big MIDI corpus via lazily-loaded sharded
piano-roll tensors instead of loading every PNG into RAM. The PNG-backed
``AnchorDataset``/``PRPairDataset``/``ShiftedTripletDataset`` and the
preencode-chunk classes are dropped: the pianoroll cache
(``baselines/data/pianoroll_cache.py``) only ever produces the sharded ``.pt``
layout these two classes read.
"""

from __future__ import annotations

import json
import os
import random
from glob import glob

import numpy as np
import torch
from scipy.ndimage import label
from scipy.stats import truncnorm
from torch.utils.data import Dataset, Sampler


def shift_no_wrap(x, shifts, dims):
    """Drop-in replacement for torch.roll with zero-fill instead of wrap."""
    if shifts == 0: return x
    out = torch.roll(x, shifts=shifts, dims=dims)
    n = abs(shifts)
    if shifts > 0: out.narrow(dims, 0, n).zero_()
    else: out.narrow(dims, out.size(dims) - n, n).zero_()
    return out


def sample_shift(max_shift, sigma=7, size=None):
    "Samples shift amounts as integers from a truncated normal distribution."
    a, b = -max_shift / sigma, max_shift / sigma
    samples = truncnorm.rvs(a, b, loc=0, scale=sigma, size=size)
    if size is None: return int(round(samples))
    return np.rint(samples).astype(int)


def sample_shifts(max_x, max_y, sigma):
    "Get X and Y shifts; one of them must be non-zero."
    while True:
        sx, sy = sample_shift(max_x, sigma), sample_shift(max_y, sigma)
        if sx != 0 or sy != 0: return sx, sy


def note_length_weights(img, min_weight=1.0, power=0.5):
    "Note weight is inversely proportional to note length."
    weights = np.ones_like(img, dtype=np.float32)
    lengths = []
    for row in range(img.shape[0]):
        labeled, n = label(img[row] > 0)
        for i in range(1, n + 1):
            run = (labeled == i)
            length = run.sum()
            lengths.append(length)
            weights[row, run] = (1.0 / length) ** power
    if lengths:
        median_length = np.median(lengths)
        weights = weights * (median_length) ** power
    return weights


class ChunkShuffleSampler(Sampler):
    """Shuffle chunk (shard) order each epoch; access items within each chunk
    sequentially, so DataLoader workers don't thrash between shards."""
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle

    def __len__(self): return len(self.dataset)

    def __iter__(self):
        n = len(self.dataset.files)
        order = torch.randperm(n).tolist() if self.shuffle else list(range(n))
        indices = []
        for ci in order:
            s, e = self.dataset.chunk_sample_ranges[ci]
            indices.extend(range(s, e))
        return iter(indices)


class _ShardCache:
    "1-slot cache so sequential (shard-shuffled) access doesn't reload a shard per sample."
    def __init__(self):
        self.idx, self.data = None, None

    def get(self, shard_paths, i):
        if self.idx != i:
            self.data = torch.load(shard_paths[i], map_location='cpu', weights_only=False)
            self.idx = i
        return self.data


def _build_shard_index(shard_dir, split):
    "Build a (shard_idx, record_idx) index from sidecar .idx.json files."
    shard_paths = sorted(glob(os.path.join(os.path.expanduser(shard_dir), f'{split}_shard*.pt')))
    assert shard_paths, f"No {split}_shard*.pt files found in {shard_dir}"
    index, ranges = [], {}
    for si, sp in enumerate(shard_paths):
        idx_path = sp.replace('.pt', '.idx.json')
        with open(idx_path) as f:
            n = json.load(f)['count']
        start = len(index)
        index.extend((si, ri) for ri in range(n))
        ranges[si] = (start, len(index))
    return shard_paths, index, ranges


class ShardedAnchorDataset(Dataset):
    """Lazy-loading, disk-backed piano-roll dataset for large-scale MIDI sets.
    Backed by sharded .pt files of uint8 (128, T) binary piano-roll arrays.
    Use with ChunkShuffleSampler so DataLoader workers visit one shard at a time.
    """
    def __init__(self, shard_dir, crop_size=128, split='train', verbose=True,
                 aug_y_max=12, sigma=7, pad_x=(0, 0)):
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        self.aug_y_max, self.sigma, self.pad_x = aug_y_max, sigma, pad_x
        self.files, self.index, self.chunk_sample_ranges = _build_shard_index(shard_dir, split)
        self._cache = _ShardCache()
        if verbose:
            print(f"ShardedAnchorDataset[{split}]: {len(self.files)} shards, {len(self.index)} records")

    def __len__(self): return len(self.index)

    def _get_array(self, idx):
        si, ri = self.index[idx]
        shard = self._cache.get(self.files, si)
        return shard[ri]

    def __getitem__(self, idx, pad_x=None, crop_size=None):
        if crop_size is None: crop_size = self.crop_size
        if isinstance(crop_size, int): crop_size = (crop_size, crop_size)
        if pad_x is None: pad_x = self.pad_x

        arr = self._get_array(idx)
        img = torch.from_numpy(arr.astype(np.float32))
        h, w = img.shape
        need = crop_size[1] + pad_x[0] + pad_x[1]
        if w < need + 1:
            img = torch.cat([img, torch.zeros(h, need + 1 - w)], dim=1)
            w = img.shape[1]

        aug_y = sample_shift(self.aug_y_max, self.sigma)
        img = shift_no_wrap(img, shifts=aug_y, dims=0)

        min_loc, max_loc = 0 + pad_x[0], w - crop_size[1] - pad_x[1]
        loc = random.randint(min_loc, max_loc)
        img = img[:, loc - pad_x[0]: loc + crop_size[1] + pad_x[1]]

        if crop_size[0] < img.shape[0]:
            mid, hc = img.shape[0] // 2, crop_size[0] // 2
            img = img[mid - hc: mid + hc, :]

        note_weights = torch.from_numpy(note_length_weights(img.numpy()).astype(np.float16))
        return {
            'img': img.unsqueeze(0),
            'file_idx': idx,
            'note_weights': note_weights.unsqueeze(0),
        }


class ShardedTripletDataset(ShardedAnchorDataset):
    """Lazy-loading, disk-backed equivalent of a shifted-triplet dataset (pitch/time
    factorization triplets) for large-scale MIDI sets."""
    def __init__(self, shard_dir, max_shift_x=12, max_shift_y=12, shared=None, aug_y_max=6, **kwargs):
        super().__init__(shard_dir, aug_y_max=aug_y_max, **kwargs)
        self.max_shift_x, self.max_shift_y, self.shared = max_shift_x, max_shift_y, shared

    def __getitem__(self, idx, requested_scheme=None, requested_target=None):
        msx = self.shared['training']['max_shift_x'] if self.shared else self.max_shift_x
        msy = self.shared['training']['max_shift_y'] if self.shared else self.max_shift_y
        while True:
            scheme = torch.randint(0, 3, (1,)).item() if requested_scheme is None else int(requested_scheme)
            if requested_target is not None and requested_target != 0: scheme = torch.randint(0, 2, (1,)).item()
            sign1 = 1 if torch.rand(1) > 0.5 else -1
            if requested_target is None:
                sign2 = 1 if torch.rand(1) > 0.5 else -1
            else:
                sign2 = sign1 * int(requested_target)
            if scheme == 0:
                s1 = abs(sample_shift(msy, self.sigma)) or 1
                s2 = abs(sample_shift(msy, self.sigma)) or 1
                dy1, dx1, dy2, dx2 = sign1 * s1, 0, sign2 * s2, 0
            elif scheme == 1:
                s1 = abs(sample_shift(msx, self.sigma)) or 1
                s2 = abs(sample_shift(msx, self.sigma)) or 1
                dy1, dx1, dy2, dx2 = 0, sign1 * s1, 0, sign2 * s2
            else:
                s1 = abs(sample_shift(msy, self.sigma)) or 1
                s2 = abs(sample_shift(msx, self.sigma)) or 1
                dy1, dx1, dy2, dx2 = sign1 * s1, 0, 0, sign2 * s2
            if not ((dy1 == dy2) and (dx1 == dx2)) and not (dx1 == 0 and dy1 == 0) and not (dx2 == 0 and dy2 == 0): break
        target = float(np.sign(sign1 * sign2)) if scheme < 2 else 0.0

        pad_x = (abs(min(min(dx1, dx2), 0)), max(max(dx1, dx2), 0))
        anchor = super().__getitem__(idx, pad_x=pad_x)
        img = anchor['img']
        cs = self.crop_size[1]

        c1 = img[:, :, pad_x[0] + dx1: pad_x[0] + dx1 + cs]
        c2 = img[:, :, pad_x[0] + dx2: pad_x[0] + dx2 + cs]
        img = img[:, :, pad_x[0]: pad_x[0] + cs]

        c1 = shift_no_wrap(c1, shifts=dy1, dims=1)
        c2 = shift_no_wrap(c2, shifts=dy2, dims=1)
        return {
            'img1': img, 'img2': c1, 'img3': c2,
            'deltas': torch.tensor([[dy1, dx1], [dy2, dx2]], dtype=torch.int),
            'file_idx': anchor['file_idx'], 'scheme': scheme, 'target': torch.tensor(target),
        }
