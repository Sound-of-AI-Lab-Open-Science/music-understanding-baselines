"""LightningDataModules wrapping midi_rae's sharded piano-roll datasets.

Two arms share one on-disk cache (``baselines/data/pianoroll_cache.py``):
``midi_rae_enc`` trains on ``ShardedTripletDataset`` triplets (pitch/time
factorization needs three views), ``midi_rae_dec`` trains on plain
``ShardedAnchorDataset`` pairs (it reconstructs from a single frozen-encoder
embedding). Both use ``ChunkShuffleSampler`` so DataLoader workers visit one
shard at a time, exactly as upstream's ``train_enc.py``/``train_dec.py`` set
their sharded dataloaders up (see ``data.format: sharded`` branch there).
"""

from __future__ import annotations

from typing import Optional

import lightning as L
from torch.utils.data import DataLoader

from baselines.models.midi_rae_lib.data import (
    ChunkShuffleSampler, ShardedAnchorDataset, ShardedTripletDataset,
)


class PianorollTripletDataModule(L.LightningDataModule):
    """Encoder-stage data: pitch/time factorization triplets."""

    def __init__(self, shard_dir: str, *, batch_size: int = 300,
                 max_shift_x: int = 12, max_shift_y: int = 12,
                 num_workers=(6, 2)):
        super().__init__()
        self.shard_dir = shard_dir
        self.batch_size = batch_size
        self.max_shift_x, self.max_shift_y = max_shift_x, max_shift_y
        nw = num_workers if isinstance(num_workers, (list, tuple)) else (num_workers, num_workers)
        self.num_workers = (int(nw[0]), int(nw[1]))
        self.train_set: Optional[ShardedTripletDataset] = None
        self.val_set: Optional[ShardedTripletDataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.train_set is not None:
            return
        self.train_set = ShardedTripletDataset(self.shard_dir, split='train',
                                               max_shift_x=self.max_shift_x, max_shift_y=self.max_shift_y)
        self.val_set = ShardedTripletDataset(self.shard_dir, split='val',
                                             max_shift_x=self.max_shift_x, max_shift_y=self.max_shift_y)

    def train_dataloader(self):
        nw = self.num_workers[0]
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          sampler=ChunkShuffleSampler(self.train_set, shuffle=True),
                          num_workers=nw, drop_last=True, pin_memory=True,
                          persistent_workers=(nw > 0), prefetch_factor=4 if nw > 0 else None)

    def val_dataloader(self):
        nw = self.num_workers[1]
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          sampler=ChunkShuffleSampler(self.val_set, shuffle=False),
                          num_workers=nw, drop_last=True, pin_memory=True,
                          persistent_workers=(nw > 0), prefetch_factor=4 if nw > 0 else None)


class PianorollAnchorDataModule(L.LightningDataModule):
    """Decoder-stage data: single piano-roll crops (encoded through the frozen encoder)."""

    def __init__(self, shard_dir: str, *, batch_size: int = 360, max_shift_y: int = 12,
                 num_workers=(6, 4)):
        super().__init__()
        self.shard_dir = shard_dir
        self.batch_size = batch_size
        self.max_shift_y = max_shift_y
        nw = num_workers if isinstance(num_workers, (list, tuple)) else (num_workers, num_workers)
        self.num_workers = (int(nw[0]), int(nw[1]))
        self.train_set: Optional[ShardedAnchorDataset] = None
        self.val_set: Optional[ShardedAnchorDataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.train_set is not None:
            return
        self.train_set = ShardedAnchorDataset(self.shard_dir, split='train', aug_y_max=self.max_shift_y)
        self.val_set = ShardedAnchorDataset(self.shard_dir, split='val', aug_y_max=self.max_shift_y)

    def train_dataloader(self):
        nw = self.num_workers[0]
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          sampler=ChunkShuffleSampler(self.train_set, shuffle=True),
                          num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=(nw > 0))

    def val_dataloader(self):
        nw = self.num_workers[1]
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          sampler=ChunkShuffleSampler(self.val_set, shuffle=False),
                          num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=(nw > 0))
