"""Core data structures for midi-rae: PatchState and HierarchicalPatchState.

Ported verbatim from upstream ``midi_rae/core.py`` (see package __init__ for
why this is vendored rather than imported).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class PatchState:
    """Bundle of patch embeddings at a single spatial scale with their metadata.

    Attributes:
        emb: (B, N, dim) patch embeddings
        pos: (N, 2) grid coordinates (row, col) for each patch
        non_empty: (B, N) content mask -- 1 where patch has content (e.g. notes), 0 for empty
        mae_mask: (N,) MAE visibility mask -- 1=visible, 0=masked out for reconstruction
    """
    emb: torch.Tensor
    pos: torch.Tensor
    non_empty: torch.Tensor
    mae_mask: torch.Tensor

    @property
    def visible(self):
        m = self.mae_mask.bool()
        return PatchState(
            emb=self.emb[:, m],
            pos=self.pos[m],
            non_empty=self.non_empty[:, m],
            mae_mask=self.mae_mask[m],
        )

    @property
    def masked(self):
        m = ~self.mae_mask.bool()
        return PatchState(
            emb=self.emb[:, m],
            pos=self.pos[m],
            non_empty=self.non_empty[:, m],
            mae_mask=self.mae_mask[m],
        )

    @property
    def non_empty_flat(self):
        return self.non_empty.reshape(-1).bool()

    @property
    def dim(self): return self.emb.shape[-1]

    @property
    def num_patches(self): return self.emb.shape[1]

    @property
    def batch_size(self): return self.emb.shape[0]

    def to(self, device):
        return PatchState(emb=self.emb.to(device), pos=self.pos.to(device),
                          non_empty=self.non_empty.to(device), mae_mask=self.mae_mask.to(device))


@dataclass
class HierarchicalPatchState:
    """Multi-scale patch states, ordered coarsest -> finest."""
    levels: list

    def __getitem__(self, idx): return self.levels[idx]
    def __len__(self): return len(self.levels)

    @property
    def coarsest(self): return self.levels[0]

    @property
    def finest(self): return self.levels[-1]

    @property
    def num_levels(self): return len(self.levels)

    @property
    def all_emb(self):
        return torch.cat([level.emb for level in self.levels], dim=1)

    @property
    def all_non_empty(self):
        return torch.cat([level.non_empty for level in self.levels], dim=1)

    def to(self, device):
        return HierarchicalPatchState(levels=[x.to(device) for x in self.levels])


@dataclass
class EncoderOutput:
    """Full encoder output.

    Attributes:
        patches: Encoded representations (visible patches only)
        full_pos: (N_full, 2) all grid positions before MAE masking (needed by decoder)
        full_non_empty: (B, N_full) all content masks before MAE masking
        mae_mask: (N_full,) the MAE mask applied (1=visible, 0=masked)
    """
    patches: HierarchicalPatchState
    full_pos: torch.Tensor
    full_non_empty: torch.Tensor
    mae_mask: torch.Tensor

    def to(self, device):
        return EncoderOutput(patches=self.patches.to(device), full_pos=self.full_pos.to(device),
                             full_non_empty=self.full_non_empty.to(device), mae_mask=self.mae_mask.to(device))
