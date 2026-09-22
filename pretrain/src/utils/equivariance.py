"""Two-view shift-equivariance loss for OctupleMIDI, after MIDI-RAE-JEPA.

Hawley, "MIDI-RAE-JEPA: Hierarchical Representation Learning and Generation for
Symbolic Music", arXiv:2607.14537 (2026), section 2.3. That system builds two views
by shifting a piano-roll crop in pitch and time and asks the embedding distance to
grow with the shift magnitude:

    L_equiv = ( ||z1 - z2||  -  alpha * sqrt(d) * ||delta_hat|| ) ** 2

The target both attracts (small shift -> close embeddings) and repels (large shift
-> far), which is why it does not need a separate anti-collapse term to keep the
two views apart.

Three deliberate departures from the paper, each because OctupleMIDI is not a
piano roll:

  * The shift moves FIELD VALUES, not content through a receptive field. A bar
    shift changes the bar field; the token index is untouched. This is the honest
    limit of the token analogue and is why bar shifts are whole-bar only -- shifting
    the position field would need time-signature-dependent carry.
  * lambda, lambda_MEP and alpha are absent from the paper text. alpha is calibrated
    here on a short smoke run so that alpha*sqrt(d) matches the observed median
    ||z1 - z2|| at maximum shift; anything else would be inventing the paper's
    numbers.
  * Drums (instrument 128) are never transposed, matching this repo's existing
    `mask_transposition`.

The pitch mechanics are lifted from src/data/masking.py::mask_transposition so the
two agree on what "melodic" means and on clamping.
"""
from __future__ import annotations

import torch

# OctupleMIDI field order, from src/data/masking.py::FIELD_NAMES
BAR, POSITION, INSTRUMENT, PITCH = 0, 1, 2, 3
DRUM_INSTRUMENT = 128
MAX_MELODIC_PITCH = 127
MAX_BAR = 255


def sample_shifts(batch_size: int, max_semitones: int, max_bars: int,
                  generator: torch.Generator, device) -> tuple:
    """|delta| ~ Beta(2,2) * max, sign uniform, per the paper's scheme."""
    def beta22(n):
        # Beta(2,2) via the order statistic: median of 3 uniforms.
        u = torch.rand(n, 3, generator=generator, device=device)
        return u.median(dim=1).values

    sgn_p = torch.where(torch.rand(batch_size, generator=generator, device=device) < 0.5,
                        -1.0, 1.0)
    sgn_b = torch.where(torch.rand(batch_size, generator=generator, device=device) < 0.5,
                        -1.0, 1.0)
    dp = (beta22(batch_size) * max_semitones * sgn_p).round().long()
    db = (beta22(batch_size) * max_bars * sgn_b).round().long()
    return dp, db


def shift_view(tokens: torch.Tensor, pad_mask: torch.Tensor,
               dp: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
    """Apply per-sample pitch and bar shifts. tokens [B,S,8] long.

    Clamping distorts near the edges, so the sign is flipped toward whichever
    direction has headroom rather than saturating a whole batch against 0 or 127.
    """
    out = tokens.clone()
    B = out.shape[0]

    melodic = (out[..., INSTRUMENT] != DRUM_INSTRUMENT) & pad_mask
    pitch = out[..., PITCH]
    lo = torch.where(melodic, pitch, torch.full_like(pitch, MAX_MELODIC_PITCH)).amin(1)
    hi = torch.where(melodic, pitch, torch.zeros_like(pitch)).amax(1)
    dp = torch.where((dp < 0) & (lo + dp < 0), -dp, dp)
    dp = torch.where((dp > 0) & (hi + dp > MAX_MELODIC_PITCH), -dp, dp)
    new_pitch = (pitch + dp.view(B, 1)).clamp(0, MAX_MELODIC_PITCH)
    out[..., PITCH] = torch.where(melodic, new_pitch, pitch)

    bar = out[..., BAR]
    blo = torch.where(pad_mask, bar, torch.full_like(bar, MAX_BAR)).amin(1)
    bhi = torch.where(pad_mask, bar, torch.zeros_like(bar)).amax(1)
    db = torch.where((db < 0) & (blo + db < 0), -db, db)
    db = torch.where((db > 0) & (bhi + db > MAX_BAR), -db, db)
    new_bar = (bar + db.view(B, 1)).clamp(0, MAX_BAR)
    out[..., BAR] = torch.where(pad_mask, new_bar, bar)

    return out, dp, db


def masked_mean_pool(h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    m = pad_mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


def equivariance_loss(z1: torch.Tensor, z2: torch.Tensor,
                      dp: torch.Tensor, db: torch.Tensor,
                      max_semitones: int, max_bars: int,
                      alpha: float) -> tuple:
    """(||z1 - z2|| - alpha*sqrt(d)*||delta_hat||)^2 / d, per sample then mean.

    z1 is the unshifted EMA target view and carries no gradient; z2 is the shifted
    student view. Returns (loss, observed_distance, target_distance) so a smoke run
    can calibrate alpha by comparing the two.
    """
    d = z1.shape[-1]
    dist = (z1.detach() - z2).norm(dim=-1)
    delta = torch.stack([dp.float() / max(max_semitones, 1),
                         db.float() / max(max_bars, 1)], dim=-1).norm(dim=-1)
    target = alpha * (d ** 0.5) * delta
    return ((dist - target) ** 2).mean() / d, dist.detach(), target.detach()
