"""EMA teacher wrapper, ported from upstream ``midi_rae/utils.py::EMAModel``
(see package __init__).
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


def to_scalar(x):
    return x.item() if hasattr(x, 'item') else x


class EMAModel(nn.Module):
    """Exponential moving average wrapper for a stable teacher-student pair."""
    def __init__(self, model, eta=0.99, update_every=1, dtype=torch.float32):
        super().__init__()
        self.eta, self.update_every, self.dtype = eta, update_every, dtype
        self.register_buffer('_steps', torch.tensor(0))
        self.ema = deepcopy(model).to(dtype)
        for p in self.ema.parameters(): p.requires_grad_(False)

    def update(self, model):
        self._steps += 1
        if self._steps % self.update_every != 0: return
        with torch.no_grad():
            for p_ema, p in zip(self.ema.parameters(), model.parameters()):
                p_ema.data.mul_(self.eta).add_(p.data.to(p_ema.data), alpha=1 - self.eta)
            for b_ema, b in zip(self.ema.buffers(), model.buffers()):
                b_ema.data.copy_(b.data.to(b_ema.data))

    def forward(self, x, **kwargs):
        with torch.no_grad():
            return self.ema(x, **kwargs)
