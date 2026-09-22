"""Fundamental Music Embedding (FME).

Faithful PyTorch port of the *Fundamental Music Embedding* from Guo, Kang &
Herremans, "A Domain-Knowledge-Inspired Music Embedding Space and a Novel
Attention Mechanism for Symbolic Music Modeling" (AAAI 2023).  The upstream
implementation is ``model/FME_music_positional_encoding.py`` (class
``Fundamental_Music_Embedding``) in
<https://github.com/guozixunnicolas/FundamentalMusicEmbedding> (pin
``793e300``).  That repository publishes **no licence**, so nothing is copied
from it: this module is an independent implementation written from the paper
and verified to agree numerically with the upstream formulation, plus the
OctupleMIDI id -> physical-value conversions the JEPA paper needs.

Why we need this
----------------
The JEPA paper (Hachana & Rasheed) fuses the octuple embedding with FME on the
Pitch and Duration attributes::

    Emb(x_i) = W_E [ Oct(x_i) ; FME_pitch(x_i) ; FME_dur(x_i) ] + PE(x_i)

FME is a *bias-adjusted sinusoidal* encoding of a scalar musical attribute.  For
a scalar value ``f`` and model width ``d`` it produces (verified line-by-line
against the upstream ``__call__`` and the upstream tutorial's ``w_k`` / ``p_k``
reference)::

    w_k          = base ** (-2k/d)                    for k in [0, d/2)
    FME(f)[2k]   = sin(w_k * f) + bias[2k]
    FME(f)[2k+1] = cos(w_k * f) + bias[2k+1]

where ``bias`` is a learnable per-dimension "translation bias" (the ``nd``
translation-bias type in the upstream code).  ``base`` and ``bias`` are the only
non-trivial knobs; ``base`` controls the frequency spread and ``bias`` shifts
the whole manifold so the space is no longer centred on the origin (the paper's
"bias-adjusted" property).

Official defaults (from upstream's ``training/config/ripo_transformer.yaml``
-- the only published FME config, used for both the pitch and duration FME of
the RIPO transformer):

    pitch FME : d_model=256, base=9919, if_trainable=True, translation_bias="nd"
    dur   FME : d_model=256, base=7920, if_trainable=True, translation_bias="nd"

The JEPA paper does NOT specify ``d_fme``; we therefore expose everything as
config and default to the official ``d_model=256`` and the official per-attribute
bases (9919 / 7920).  See ``configs/jepa_data.yaml``.

OctupleMIDI id -> physical value
--------------------------------
The octuple *field values* are quantized bin indices, not physical quantities.
FME wants the physical value so that its translation-invariance property is
musically meaningful (a +2 shift of the FME input == a whole-tone transposition,
etc.).  We invert the quantization exactly as ``src.data.octuple`` defines it:

* Pitch field value ``v`` (0..255):
    - melodic note (``v <= 127``)  -> MIDI pitch ``v``            (identity)
    - drum note   (``v >= 128``)   -> MIDI key   ``v - 128``
  (``octuple`` stores drums as ``note.pitch + MAX_PITCH + 1``; we invert that.
  For drums the resulting number is a percussion *key*, not a tonal pitch -- see
  the design note in the report; the instrument field, fused separately, already
  distinguishes drum from melodic notes.)

* Duration field value ``c`` (0..127) is a ``d2e`` code; the physical duration
  in *positions* (1/POS_RESOLUTION beat) is ``e2d(c)`` and in *beats* is
  ``e2d(c) / POS_RESOLUTION``.  We default to **beats** (a natural, O(1)-scale
  unit for the sinusoid); ``duration_unit="pos"`` selects raw positions.

This module has no numpy dependency and is import-safe before the full env is
installed (it only needs ``torch`` and the pure-Python ``octuple`` tables).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.data import octuple as oct

# ---------------------------------------------------------------------------
# Official defaults (see module docstring).  Paper (Hachana & Rasheed) does NOT
# specify these; values come from the upstream RIPO transformer config.
# ---------------------------------------------------------------------------
DEFAULT_D_FME = 256
DEFAULT_PITCH_BASE = 9919
DEFAULT_DURATION_BASE = 7920

MAX_MELODIC_PITCH = oct.MAX_PITCH            # 127
DRUM_PITCH_OFFSET = oct.MAX_PITCH + 1        # 128 (octuple adds this for drums)


class FundamentalMusicEmbedding(nn.Module):
    """Bias-adjusted sinusoidal embedding of a scalar attribute.

    Numerically identical to the upstream ``Fundamental_Music_Embedding`` forward
    pass.  Operates on *physical* scalar values (e.g. MIDI pitch, duration in
    beats), NOT on quantized ids -- wrap it with :class:`AttributeFME` (or use
    :func:`build_pitch_fme` / :func:`build_duration_fme`) to feed octuple field
    values directly.

    Parameters
    ----------
    d_model:
        Embedding width ``d`` (must be even; the sinusoid pairs dims).
    base:
        Frequency base ``B``.  ``w_k = base ** (-2k/d)``.
    if_trainable:
        If True the sinusoid frequencies (``angles``) are a learnable
        ``nn.Parameter`` initialised to the analytic rates (matches the RIPO
        config, ``if_trainable=True``); else a fixed buffer.
    translation_bias:
        ``"nd"`` -> learnable per-dimension bias of shape ``(1, d_model)``
        (upstream default, initialised ``U[0,1)``); ``None`` -> no bias.
    translation_bias_trainable:
        Whether the bias is learnable (upstream default True).
    """

    def __init__(
        self,
        d_model: int = DEFAULT_D_FME,
        base: float = 10000.0,
        if_trainable: bool = True,
        translation_bias: Optional[str] = "nd",
        translation_bias_trainable: bool = True,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("FME d_model must be even, got {}".format(d_model))
        self.d_model = int(d_model)
        self.base = float(base)

        # angle_rates[i] = 1 / base ** (2 * (i // 2) / d_model)   (upstream formula)
        i = torch.arange(self.d_model, dtype=torch.float32)
        angle_rates = 1.0 / torch.pow(torch.tensor(self.base), (2 * (i // 2)) / self.d_model)
        angle_rates = angle_rates.unsqueeze(0)  # (1, d_model)
        if if_trainable:
            self.angles = nn.Parameter(angle_rates, requires_grad=True)
        else:
            self.register_buffer("angles", angle_rates)
        self.if_trainable = bool(if_trainable)

        # Translation bias (the "bias-adjusted" part).  Upstream "nd" == per-dim.
        self.translation_bias_type = translation_bias
        if translation_bias is None:
            self.translation_bias = None
        else:
            if translation_bias == "2d":
                bias = torch.rand((1, 2), dtype=torch.float32)
            elif translation_bias == "nd":
                bias = torch.rand((1, self.d_model), dtype=torch.float32)
            else:
                raise ValueError("translation_bias must be 'nd', '2d' or None")
            if translation_bias_trainable:
                self.translation_bias = nn.Parameter(bias, requires_grad=True)
            else:
                self.register_buffer("translation_bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed real-valued scalars.

        Parameters
        ----------
        x: ``[B, N]`` or ``[N]`` real-valued tensor of attribute values.

        Returns
        -------
        ``[B, N, d_model]`` (a leading batch axis is added for 1-D input).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, N)
        x = x.to(self.angles.dtype)
        angle_rads = x.unsqueeze(-1) * self.angles  # (B, N, d_model)

        out = torch.empty_like(angle_rads)
        out[..., 0::2] = torch.sin(angle_rads[..., 0::2])  # even dims -> sin
        out[..., 1::2] = torch.cos(angle_rads[..., 1::2])  # odd dims  -> cos

        if self.translation_bias is not None:
            bias = self.translation_bias
            if bias.shape[-1] != self.d_model:  # "2d" bias -> broadcast/repeat (upstream)
                bias = bias.repeat(1, self.d_model // 2)
            out = out + bias
        return out

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return "d_model={}, base={}, trainable_angles={}, translation_bias={}".format(
            self.d_model, self.base, self.if_trainable, self.translation_bias_type
        )


# ---------------------------------------------------------------------------
# OctupleMIDI id -> physical value conversions (see module docstring).
# ---------------------------------------------------------------------------
# e2d lookup: duration code -> position count.  Registered as a buffer on the
# wrapper so device moves are automatic.
_DUR_DEC = torch.tensor(oct.dur_dec, dtype=torch.float32)  # len == 128


def pitch_value_to_midi(pitch_value: torch.Tensor) -> torch.Tensor:
    """Octuple pitch field value (0..255) -> MIDI number (0..127).

    Melodic notes are the identity; drum notes (>=128) are mapped back to their
    stored MIDI key by subtracting the ``MAX_PITCH+1`` offset octuple added.
    """
    return torch.where(
        pitch_value > MAX_MELODIC_PITCH,
        pitch_value - DRUM_PITCH_OFFSET,
        pitch_value,
    ).to(pitch_value.dtype)


def duration_value_to_physical(
    dur_value: torch.Tensor, unit: str = "beat"
) -> torch.Tensor:
    """Octuple duration code (0..127) -> physical duration.

    ``unit="beat"`` (default) returns duration in beats; ``unit="pos"`` returns
    it in positions (1/POS_RESOLUTION beat).  Uses the exact ``e2d`` table.
    """
    dur_dec = _DUR_DEC.to(dur_value.device)
    idx = dur_value.long().clamp(0, dur_dec.numel() - 1)
    positions = dur_dec[idx]
    if unit == "pos":
        return positions
    if unit == "beat":
        return positions / oct.POS_RESOLUTION
    raise ValueError("unit must be 'beat' or 'pos'")


class AttributeFME(nn.Module):
    """FME for a single OctupleMIDI attribute, taking *field values* as input.

    This is the module the JEPA paper's ``FME_pitch`` / ``FME_dur`` map to and
    it satisfies the requested interface ``fme(field_ids) -> [B, N, d_fme]``.
    It converts the quantized octuple field value to a physical quantity (via
    :func:`pitch_value_to_midi` / :func:`duration_value_to_physical`) and then
    applies the underlying :class:`FundamentalMusicEmbedding`.

    ``attribute`` is ``"pitch"`` or ``"duration"``.
    """

    def __init__(
        self,
        attribute: str,
        d_model: int = DEFAULT_D_FME,
        base: Optional[float] = None,
        if_trainable: bool = True,
        translation_bias: Optional[str] = "nd",
        translation_bias_trainable: bool = True,
        duration_unit: str = "beat",
    ) -> None:
        super().__init__()
        if attribute not in ("pitch", "duration"):
            raise ValueError("attribute must be 'pitch' or 'duration'")
        self.attribute = attribute
        self.duration_unit = duration_unit
        if base is None:
            base = DEFAULT_PITCH_BASE if attribute == "pitch" else DEFAULT_DURATION_BASE
        self.fme = FundamentalMusicEmbedding(
            d_model=d_model,
            base=base,
            if_trainable=if_trainable,
            translation_bias=translation_bias,
            translation_bias_trainable=translation_bias_trainable,
        )
        # e2d table as a buffer so ``.to(device)`` moves it.
        if attribute == "duration":
            self.register_buffer("dur_dec", _DUR_DEC.clone())

    @property
    def d_model(self) -> int:
        return self.fme.d_model

    def field_value_to_physical(self, field_ids: torch.Tensor) -> torch.Tensor:
        if self.attribute == "pitch":
            return pitch_value_to_midi(field_ids)
        idx = field_ids.long().clamp(0, self.dur_dec.numel() - 1)
        positions = self.dur_dec[idx]
        if self.duration_unit == "pos":
            return positions
        return positions / oct.POS_RESOLUTION

    def forward(self, field_ids: torch.Tensor) -> torch.Tensor:
        """``field_ids``: ``[B, N]`` or ``[N]`` octuple field values -> ``[B, N, d]``."""
        physical = self.field_value_to_physical(field_ids)
        return self.fme(physical)


def build_pitch_fme(d_model: int = DEFAULT_D_FME, base: float = DEFAULT_PITCH_BASE,
                    **kwargs) -> AttributeFME:
    """Pitch FME with official defaults (d=256, base=9919)."""
    return AttributeFME("pitch", d_model=d_model, base=base, **kwargs)


def build_duration_fme(d_model: int = DEFAULT_D_FME, base: float = DEFAULT_DURATION_BASE,
                       duration_unit: str = "beat", **kwargs) -> AttributeFME:
    """Duration FME with official defaults (d=256, base=7920)."""
    return AttributeFME("duration", d_model=d_model, base=base,
                        duration_unit=duration_unit, **kwargs)


class OctupleFME(nn.Module):
    """Convenience holder for the JEPA input: pitch FME + duration FME.

    Given an octuple *field-value* tensor ``[B, N, 8]`` (the representation
    emitted by :mod:`src.data.masking` / :mod:`src.data.jepa_datamodule`), returns
    the concatenation ``[FME_pitch ; FME_dur]`` of shape ``[B, N, 2*d_fme]`` (or
    the two tensors separately via :meth:`forward_split`).  The pitch column is
    field index 3, the duration column is field index 4 (octuple order
    ``bar,pos,instrument,pitch,duration,velocity,timesig,tempo``).
    """

    PITCH_COL = 3
    DURATION_COL = 4

    def __init__(
        self,
        d_fme: int = DEFAULT_D_FME,
        pitch_base: float = DEFAULT_PITCH_BASE,
        duration_base: float = DEFAULT_DURATION_BASE,
        if_trainable: bool = True,
        translation_bias: Optional[str] = "nd",
        duration_unit: str = "beat",
    ) -> None:
        super().__init__()
        self.d_fme = d_fme
        self.pitch_fme = build_pitch_fme(
            d_model=d_fme, base=pitch_base, if_trainable=if_trainable,
            translation_bias=translation_bias,
        )
        self.duration_fme = build_duration_fme(
            d_model=d_fme, base=duration_base, if_trainable=if_trainable,
            translation_bias=translation_bias, duration_unit=duration_unit,
        )

    def forward_split(self, octuple_values: torch.Tensor):
        """Return ``(fme_pitch, fme_dur)``, each ``[B, N, d_fme]``."""
        pitch_ids = octuple_values[..., self.PITCH_COL]
        dur_ids = octuple_values[..., self.DURATION_COL]
        return self.pitch_fme(pitch_ids), self.duration_fme(dur_ids)

    def forward(self, octuple_values: torch.Tensor) -> torch.Tensor:
        fme_p, fme_d = self.forward_split(octuple_values)
        return torch.cat([fme_p, fme_d], dim=-1)
