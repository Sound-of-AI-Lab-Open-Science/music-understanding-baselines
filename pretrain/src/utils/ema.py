"""Exponential moving average (EMA) for the Music-JEPA target encoder.

The target encoder ``f_theta_EMA`` is not trained by gradient descent; its
weights track the online (context) encoder ``f_theta`` via

    theta_EMA <- rho * theta_EMA + (1 - rho) * theta

applied *once per optimizer step* (I-JEPA / BYOL convention).  ``rho`` follows a
schedule from ``rho_start`` (default 0.996) up to ``rho_end`` (default 1.0) over
the course of training, so the target moves quickly early on and freezes as the
online network converges.

This module provides three composable pieces:

* :func:`update_ema` -- the in-place update primitive (params **and** buffers),
  wrapped in ``no_grad``.
* :class:`EmaMomentumSchedule` -- maps a global step to ``rho`` (linear or
  cosine interpolation between ``rho_start`` and ``rho_end``).
* :class:`EmaCallback` -- an optional PyTorch-Lightning ``Callback`` that drives
  the update from ``on_before_zero_grad`` (fires exactly once per optimizer
  step, so gradient accumulation is handled correctly).  The LightningModule may
  instead call :func:`update_ema` itself; both paths are supported.

The primitive is deliberately framework-agnostic (plain ``nn.Module`` args) so
it is unit-testable without Lightning.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:  # Lightning is optional at import time (keeps the primitive importable).
    import lightning.pytorch as pl
    _CallbackBase = pl.Callback
    _HAS_PL = True
except Exception:  # pragma: no cover
    try:
        import pytorch_lightning as pl  # type: ignore
        _CallbackBase = pl.Callback
        _HAS_PL = True
    except Exception:  # pragma: no cover
        _HAS_PL = False

        class _CallbackBase:  # type: ignore
            pass


@torch.no_grad()
def update_ema(target: nn.Module, online: nn.Module, rho: float) -> None:
    """In-place ``target <- rho * target + (1 - rho) * online`` for a module pair.

    Updates both parameters and buffers so running statistics (if any) also
    track.  ``target`` must be an architectural copy of ``online`` (same
    ``state_dict`` keys and shapes); this holds when ``target`` is created via
    ``copy.deepcopy(online)``.  The update runs under ``no_grad`` and mutates
    ``target`` only -- no gradient ever flows to it.
    """
    rho = float(rho)
    one_minus = 1.0 - rho
    t_params = dict(target.named_parameters())
    o_params = dict(online.named_parameters())
    for name, op in o_params.items():
        tp = t_params[name]
        # tp = rho*tp + (1-rho)*op  (done in fp32-safe in-place arithmetic)
        tp.mul_(rho).add_(op.detach(), alpha=one_minus)

    t_bufs = dict(target.named_buffers())
    o_bufs = dict(online.named_buffers())
    for name, ob in o_bufs.items():
        tb = t_bufs.get(name)
        if tb is None or not torch.is_floating_point(tb):
            # integer / non-float buffers (e.g. counters, masks) are copied
            # verbatim so they stay consistent with the online net.
            if tb is not None:
                tb.copy_(ob)
            continue
        tb.mul_(rho).add_(ob, alpha=one_minus)


class EmaMomentumSchedule:
    """``rho(step)`` interpolating ``rho_start -> rho_end`` over ``total_steps``.

    ``mode="linear"`` (I-JEPA default) or ``"cosine"``.  Steps past
    ``total_steps`` clamp to ``rho_end``.  ``total_steps`` may be set later via
    :meth:`set_total_steps` (the number of optimizer steps is often only known
    once the trainer/datamodule are built).
    """

    def __init__(
        self,
        rho_start: float = 0.996,
        rho_end: float = 1.0,
        total_steps: Optional[int] = None,
        mode: str = "linear",
    ) -> None:
        if mode not in ("linear", "cosine"):
            raise ValueError("mode must be 'linear' or 'cosine', got {!r}".format(mode))
        self.rho_start = float(rho_start)
        self.rho_end = float(rho_end)
        self.total_steps = int(total_steps) if total_steps else None
        self.mode = mode

    def set_total_steps(self, total_steps: int) -> None:
        self.total_steps = max(1, int(total_steps))

    def value(self, step: int) -> float:
        if not self.total_steps or self.total_steps <= 1:
            return self.rho_start
        frac = min(max(step / float(self.total_steps), 0.0), 1.0)
        if self.mode == "cosine":
            # ease-in-out; both endpoints have zero slope
            frac = 0.5 * (1.0 - math.cos(math.pi * frac))
        return self.rho_start + (self.rho_end - self.rho_start) * frac


class EmaCallback(_CallbackBase):
    """Lightning callback updating a target encoder after every optimizer step.

    Expects the ``LightningModule`` to expose ``target_encoder`` and
    ``context_encoder`` attributes (the online/target pair) and, optionally, an
    ``ema_schedule`` (:class:`EmaMomentumSchedule`); if absent, a schedule is
    built from the constructor arguments.  Uses ``on_before_zero_grad`` which
    Lightning calls exactly once per optimizer step (respecting
    ``accumulate_grad_batches``).

    The LightningModule in :mod:`src.tasks.jepa_pretrain` performs the EMA update
    itself, so this callback is provided mainly for reuse / alternative wiring.
    """

    def __init__(
        self,
        rho_start: float = 0.996,
        rho_end: float = 1.0,
        total_steps: Optional[int] = None,
        mode: str = "linear",
    ) -> None:
        self.schedule = EmaMomentumSchedule(rho_start, rho_end, total_steps, mode)

    def on_fit_start(self, trainer, pl_module):  # pragma: no cover - lightning hook
        if self.schedule.total_steps is None:
            try:
                self.schedule.set_total_steps(int(trainer.estimated_stepping_batches))
            except Exception:
                pass

    def on_before_zero_grad(self, trainer, pl_module, optimizer):  # pragma: no cover
        sched = getattr(pl_module, "ema_schedule", self.schedule)
        rho = sched.value(int(trainer.global_step))
        update_ema(pl_module.target_encoder, pl_module.context_encoder, rho)
        pl_module.log("train/ema_rho", rho, on_step=True, on_epoch=False, prog_bar=False)
