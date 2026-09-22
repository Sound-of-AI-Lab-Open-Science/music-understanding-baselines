"""Music-JEPA model: context/target encoders + latent-conditioned predictor.

Faithful-as-possible implementation of the Joint-Embedding Predictive
Architecture of Hachana & Rasheed (*Using a Joint-Embedding Predictive
Architecture for Symbolic Music Understanding*), which has no public code.  The
paper fixes the high-level recipe (§3.2-3.3); everything the paper leaves open is
made an explicit, documented, configurable default (see :class:`MusicJepaConfig`
and ``configs/jepa_pretrain_*.yaml``).

Pipeline (one training step)::

    context_tokens [B,S,8] --Emb--> f_theta      --> x  [B,S,d]   (context repr)
    target_tokens  [B,S,8] --Emb--> f_theta_EMA  --> y  [B,S,d]   (sg, EMA, eval)
                              predictor(x, latent, target_pos) --> yhat [B,T,d]
    loss = alpha*L_cos(yhat, y[target_pos]) + beta*L_var(yhat) + sigma*L_cov(yhat)

Input embedding (paper §3.2, exactly)::

    Oct(x_i) = W_O [ E_1(x_{i,1}) ; ... ; E_8(x_{i,8}) ]        (8 field embeddings)
    Emb(x_i) = W_E [ Oct(x_i) ; FME_pitch(x_i) ; FME_dur(x_i) ] + PE(x_i)

with per-field embedding tables sized to the octuple field-value space
(:data:`src.data.masking.FIELD_SIZE` = ``[256,128,129,256,128,32,254,49]``), the
Fundamental Music Embedding of :mod:`src.models.fme` fused on the pitch/duration
fields, and an **absolute sinusoidal** positional encoding ``PE`` (the paper
studies absolute PE explicitly in its Appendix C ablation).  The ``PE`` term is
switchable via ``cfg.pos_encoding`` (``absolute`` default / ``none`` / ``relative``
/ ``absolute_relative``).  ``relative`` reproduces the paper's "No Absolute PE
(Relative PE)" *ablation* variant; ``absolute_relative`` is the paper's actual
**main model** -- absolute index-based ``PE(x_i)`` kept *and* a relative
key-query self-attention (Huang et al. 2020, ref [13]).  Which flavour of
relative attention the relative encoders use is picked by ``cfg.rel_attn_method``
-- the true Huang et al. 2020 methods 1/2/3 (multiplicative modulation of the
content score by a scalar/vector relative term) or 4 (the paper's additive
relative-position *vectors* dotted with both query and key).

Key design decisions (paper is silent -> we choose and document)
--------------------------------------------------------------
* **Masked context positions keep full length; the hidden slots are replaced by
  a single learned ``[MASK]`` embedding** (BERT / data2vec style), *not* deleted
  as in vision I-JEPA.  This is dictated by the data contract of
  :mod:`src.data.masking` (masked slots are zero placeholders + a ``context_mask``
  telling the model which to substitute) and keeps absolute positions aligned
  between context, target and the predictor's positional queries.  The
  substitution is applied only where ``context_mask`` is False *and* the position
  is real (``pad_mask`` True); for the augmentation masks (rhythm-noise /
  transposition) ``context_mask`` is all-True so no substitution happens -- the
  context is a perturbed full view, as intended.

* **Predictor = I-JEPA predictor + latent conditioning.**  Context representations
  are linearly projected to the (narrower) predictor width, given their own
  absolute PE, and concatenated with ``T`` learned *query* tokens carrying the
  target-position PE.  The masking **latent variable** (which of the 7 methods +
  its numeric parameters) is injected as ``mask_type_embedding(type) +
  MLP(params)`` and **added to every query token**, so the predictor knows *what
  kind of corruption it must invert* -- this is the paper's extra "latent
  variable fed to the predictor" beyond vanilla I-JEPA.  A transformer mixes
  context + queries; the query outputs are projected back to the encoder width to
  produce ``yhat``.

* **Target encoder** is a ``deepcopy`` of the context encoder at init (so
  ``theta_EMA(0) = theta(0)``), permanently ``requires_grad_(False)`` and forced
  to ``eval()`` (no dropout in the target branch -- important: a stochastic
  target would make the regression objective noisy).  ``MusicJepa.train()`` is
  overridden so switching the module to train mode never re-enables the target
  encoder's dropout.

* Pre-LN transformer blocks (``norm_first=True``) with GELU -- the ViT/I-JEPA
  convention -- via ``nn.TransformerEncoder`` (``enable_nested_tensor=False`` so
  the padded-key fast path is deterministic).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch
import torch.nn as nn

from src.data.masking import FIELD_SIZE, NUM_FIELDS, NUM_MASK_TYPES, LATENT_PARAM_DIM
from src.models.fme import OctupleFME, DEFAULT_D_FME, DEFAULT_PITCH_BASE, DEFAULT_DURATION_BASE


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
@dataclass
class MusicJepaConfig:
    """Structural hyper-parameters for :class:`MusicJepa`.

    Defaults are the coordinator-fixed assumptions (paper unspecified): a 6-layer
    d=512 context/target encoder and a shallow 3-layer d=256 predictor.
    """

    # -- shared input embedding --
    d_model: int = 512
    field_embed_dim: int = 512            # per-field embedding width (concat then W_O)
    field_sizes: List[int] = field(default_factory=lambda: list(FIELD_SIZE))
    d_fme: int = DEFAULT_D_FME            # 256
    use_fme: bool = True                  # False = ablate the FME fusion entirely
    pitch_base: float = DEFAULT_PITCH_BASE
    duration_base: float = DEFAULT_DURATION_BASE
    fme_trainable: bool = True
    fme_translation_bias: Optional[str] = "nd"
    duration_unit: str = "beat"
    max_positions: int = 1024             # PE table size (>= seq_len)

    # -- positional-encoding switch (paper Appendix C ablations + main model) --
    # 'absolute'          : index-based sinusoidal PE(x_i) added to the input
    #                       embedding (the paper's Emb(x_i) = ... + PE(x_i); the
    #                       BASELINE and default -- keeps behaviour bit-for-bit).
    # 'none'              : no absolute PE and no relative bias (pure "no PE"; the
    #                       encoder sees no positional signal -- a collapse control).
    # 'relative'          : drop the absolute input PE and instead inject a learned
    #                       relative position term into every encoder self-attention
    #                       layer (see RelativeTransformerEncoder).  This is the
    #                       paper's "No Absolute PE (Relative PE)" *ablation*
    #                       variant, which the paper reports collapses.
    # 'absolute_relative' : the paper's actual MAIN model -- absolute PE(x_i) kept
    #                       (added to the input embedding, exactly as 'absolute')
    #                       *and* a relative key-query self-attention encoder.  The
    #                       absolute PE breaks the positional collapse of the pure
    #                       'relative' ablation while the relative attention supplies
    #                       the paper's §3.2 relative key-query mechanism.
    pos_encoding: str = "absolute"
    # Flavour of relative attention used by the 'relative' / 'absolute_relative'
    # encoders (ignored for 'absolute' / 'none').  All four are the true
    # Huang et al. 2020 formulations (arXiv:2009.13658), keyed by d = j - i clipped
    # to +/- rel_pos_max_distance.  Methods 1/2/3 MULTIPLY the content score q.k;
    # method 4 ADDS relative dot-products to it:
    #   1 = eq. 11-12: (q_i.k_j) * a_{|d|} / sqrt(hd)  -- per-head SCALAR, ABSOLUTE
    #       distance (sign-agnostic), init a=1 (starts at standard attention).
    #   2 = eq. 13:    (q_i.k_j) * a_{d}   / sqrt(hd)  -- per-head SCALAR, SIGNED
    #       distance.  DEFAULT.
    #   3 = eq. 15:    (sum_c q_ic k_jc a_{d,c}) / sqrt(hd)  -- per-head VECTOR
    #       (dim=head_dim) gate over the SIGNED distance; milder than method 4.
    #   4 = eq. 16:    (q_i.k_j + q_i.a_d + k_j.a_d) / sqrt(hd)  -- per-head
    #       relative-position VECTORS dotted with BOTH query and key (init a~=0).
    #       The paper's §3.2 "relative key-query attention" for the main model.
    rel_attn_method: int = 2
    # Final LayerNorm on the encoder output.  True = original behaviour (pre-LN
    # convention; pins per-dim repr variance at ~0.5-0.6).  False = free-growing
    # representation scale, as in the paper's Fig.4 healthy runs (their per-dim
    # context variance rises to ~10.5 -- impossible through a trailing LN).
    encoder_final_norm: bool = True
    rel_pos_num_buckets: int = 32         # method 2 only: #distance buckets
    rel_pos_max_distance: int = 128       # relative attn: bucket cutoff (m2) /
    #                                       clip window k (m4, gives 2k+1 vectors)

    # -- context / target encoder --
    encoder_layers: int = 6
    encoder_heads: int = 8
    encoder_ffn: int = 2048
    encoder_dropout: float = 0.1

    # -- predictor --
    predictor_dim: int = 256
    predictor_layers: int = 3
    predictor_heads: int = 4
    predictor_ffn: int = 1024             # 4 * predictor_dim
    predictor_dropout: float = 0.1

    # -- latent conditioning --
    num_mask_types: int = NUM_MASK_TYPES  # 7
    latent_param_dim: int = LATENT_PARAM_DIM  # 10

    init_std: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MusicJepaConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})


# ---------------------------------------------------------------------------
# Positional encoding.
# ---------------------------------------------------------------------------
def build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """Standard absolute sinusoidal positional encoding table ``[max_len, d]``."""
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _make_pre_ln_encoder(d_model, nhead, ffn, dropout, num_layers) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=ffn, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
        enable_nested_tensor=False,
    )


POS_ENCODING_MODES = ("absolute", "relative", "none", "absolute_relative")
# Modes whose input embedding adds the absolute index-based PE(x_i).
_ABS_PE_MODES = ("absolute", "absolute_relative")
# Modes whose encoder uses the relative-attention Transformer.
_REL_ATTN_MODES = ("relative", "absolute_relative")


# ---------------------------------------------------------------------------
# Relative position encoding (paper's "Relative PE" ablation variant).
#
# We implement the additive *learned relative position bias* family of Huang et
# al. 2020 ("Improve Transformer Models with Better Relative Position
# Embeddings", ref [13] in the paper) in its T5 bucketed form: a per-head scalar
# bias b_h(i, j) that depends only on the signed distance (j - i) is added to the
# raw attention logit q_i . k_j before the softmax.  Distances are mapped to a
# small set of buckets (exact for near neighbours, logarithmically compressed for
# far ones), so a fixed-size table generalises to any sequence length.  This is
# the standard, cheap way to give a Transformer relative positional awareness
# without any index-based absolute PE.
# ---------------------------------------------------------------------------
def _relative_position_bucket(relative_position: torch.Tensor,
                              num_buckets: int = 32,
                              max_distance: int = 128) -> torch.Tensor:
    """Map signed relative positions ``(j - i)`` to bucket ids (T5, bidirectional).

    Half the buckets encode negative offsets, half positive.  Within each half,
    the first ``num_buckets//4`` offsets are exact and the rest are placed on a
    logarithmic scale up to ``max_distance``.  Returns a long tensor shaped like
    ``relative_position``.
    """
    ret = torch.zeros_like(relative_position, dtype=torch.long)
    num_buckets //= 2
    ret += (relative_position > 0).long() * num_buckets
    n = relative_position.abs()

    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact + 1e-6)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)
    ret += torch.where(is_small, n, val_if_large)
    return ret


def _relative_clip_index(q_len: int, k_len: int, max_distance: int, device) -> torch.Tensor:
    """Clipped signed relative position ``(j - i)`` -> id in ``[0, 2*max_distance]``.

    Huang et al. 2020 *method 4* uses ``a_ij = w_{clip(j - i, k)}`` (Shaw-style):
    the signed query->key offset is clamped to ``+/-k`` (``k = max_distance``) so
    everything farther than ``k`` shares the boundary vector, then shifted by ``k``
    to a non-negative index.  Returns a long tensor ``[q_len, k_len]`` indexing the
    ``2k + 1`` learnable relative-position vectors.
    """
    ctx = torch.arange(q_len, device=device)[:, None]
    mem = torch.arange(k_len, device=device)[None, :]
    rel = (mem - ctx).clamp(-max_distance, max_distance) + max_distance   # [q, k]
    return rel


class RelativePositionBias(nn.Module):
    """Learned per-head additive relative-position bias ``[num_heads, S, S]``."""

    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.rel_emb = nn.Embedding(num_buckets, num_heads)

    def forward(self, q_len: int, k_len: int, device) -> torch.Tensor:
        ctx = torch.arange(q_len, device=device)[:, None]
        mem = torch.arange(k_len, device=device)[None, :]
        buckets = _relative_position_bucket(
            mem - ctx, self.num_buckets, self.max_distance)          # [q, k]
        values = self.rel_emb(buckets)                               # [q, k, H]
        return values.permute(2, 0, 1)                              # [H, q, k]


class RelativeMultiheadAttention(nn.Module):
    """Batched multi-head self-attention with an additive relative-position bias.

    Same maths as ``nn.MultiheadAttention`` (scaled dot-product, ``batch_first``)
    but the pre-softmax logits get ``+ rel_bias[h, i, j]``.  ``key_padding_mask``
    (``[B, S]``, True = pad) masks padded keys.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rel_bias, key_padding_mask=None):
        b, s, _ = x.shape
        h, hd = self.num_heads, self.head_dim

        def split(t):
            return t.view(b, s, h, hd).transpose(1, 2)              # [B,H,S,hd]

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,S,S]
        scores = scores + rel_bias.unsqueeze(0)                     # + [1,H,S,S]
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                                # [B,H,S,hd]
        out = out.transpose(1, 2).reshape(b, s, h * hd)
        return self.out_proj(out)


class RelativeEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer using :class:`RelativeMultiheadAttention`.

    Mirrors ``nn.TransformerEncoderLayer(norm_first=True, activation='gelu')`` so
    the 'relative' branch is architecturally identical to the 'absolute'/'none'
    branch apart from (a) no absolute input PE and (b) the relative attention bias.
    """

    def __init__(self, d_model, num_heads, ffn, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelativeMultiheadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ffn)
        self.activation = nn.GELU()
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, rel_bias, key_padding_mask=None):
        x = x + self.dropout1(self.attn(self.norm1(x), rel_bias, key_padding_mask))
        h = self.linear2(self.dropout_ffn(self.activation(self.linear1(self.norm2(x)))))
        x = x + self.dropout2(h)
        return x


# ---------------------------------------------------------------------------
# Huang et al. 2020 *method 4* relative attention (learnable relative VECTORS).
#
# The paper's §3.2 "relative key-query attention" cites Huang, Liang, Xu & Xiang
# 2020 (arXiv:2009.13658) method 4, eq. (16):
#
#     e_ij = ( q_i . k_j  +  q_i . a_ij  +  k_j . a_ij ) / sqrt(head_dim)
#
# where a_ij = w_{clip(j - i, k)} is a *learnable vector* (dimension = head_dim,
# one per clipped signed distance, per head).  Unlike method 2 (a per-head scalar
# bias added to the logit) the relative term here is a genuine vector dotted with
# BOTH the query and the key -- the extra content x position interactions.  With
# a_ij = 0 this reduces exactly to standard scaled dot-product attention.
# ---------------------------------------------------------------------------
class RelativeVectorMultiheadAttention(nn.Module):
    """Multi-head self-attention with Huang-2020 *method 4* relative vectors.

    Same projections / masking / ``batch_first`` layout as
    :class:`RelativeMultiheadAttention`, but the pre-softmax logit gains the two
    dot-product terms ``q_i . a_ij`` and ``k_j . a_ij`` (all three terms share the
    single ``1/sqrt(head_dim)`` scale, per eq. 16).  The relative vectors are
    per-head and per-layer (this module owns its own table), stored as an
    ``nn.Embedding`` so :meth:`MusicJepa._init_weights` initialises them (small
    ``init_std``) -- so training starts essentially at standard attention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float, max_distance: int = 128):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_distance = max_distance
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # per-head relative-position vectors: (2k+1) distances x (heads*head_dim),
        # viewed as [R, H, head_dim] at use time.  a_ij = rel_emb[clip(j-i)+k].
        self.rel_emb = nn.Embedding(2 * max_distance + 1, num_heads * self.head_dim)

    def forward(self, x, rel_index, key_padding_mask=None):
        b, s, _ = x.shape
        h, hd = self.num_heads, self.head_dim

        def split(t):
            return t.view(b, s, h, hd).transpose(1, 2)              # [B,H,S,hd]

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        a = self.rel_emb.weight.view(-1, h, hd).to(q.dtype)         # [R,H,hd]

        content = torch.matmul(q, k.transpose(-2, -1))              # [B,H,S,S] q.k
        # q_i . a_r  and  k_j . a_r  for every relative-distance id r
        qa = torch.einsum("bhid,rhd->bhir", q, a)                  # [B,H,S,R]
        ka = torch.einsum("bhjd,rhd->bhjr", k, a)                  # [B,H,S,R]
        # gather the per-pair relative id: q term wants a_{ij} at output (i,j);
        # k term is ka[.,.,j, id(i,j)] so gather with the transposed id then swap.
        idx = rel_index.unsqueeze(0).unsqueeze(0).expand(b, h, s, s)          # [B,H,S,S]
        idx_t = rel_index.t().unsqueeze(0).unsqueeze(0).expand(b, h, s, s)
        q_rel = torch.gather(qa, -1, idx)                          # q_i . a_ij
        k_rel = torch.gather(ka, -1, idx_t).transpose(-2, -1)      # k_j . a_ij

        scores = (content + q_rel + k_rel) * self.scale            # eq. 16
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)                                # [B,H,S,hd]
        out = out.transpose(1, 2).reshape(b, s, h * hd)
        return self.out_proj(out)


class RelativeVectorEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer using :class:`RelativeVectorMultiheadAttention`.

    Identical wiring to :class:`RelativeEncoderLayer` (so 'absolute_relative' with
    ``rel_attn_method=4`` differs from 'absolute' only in the attention's relative
    term), except the attention consumes a precomputed relative-index matrix.
    """

    def __init__(self, d_model, num_heads, ffn, dropout, max_distance=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelativeVectorMultiheadAttention(d_model, num_heads, dropout, max_distance)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ffn)
        self.activation = nn.GELU()
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, rel_index, key_padding_mask=None):
        x = x + self.dropout1(self.attn(self.norm1(x), rel_index, key_padding_mask))
        h = self.linear2(self.dropout_ffn(self.activation(self.linear1(self.norm2(x)))))
        x = x + self.dropout2(h)
        return x


# ---------------------------------------------------------------------------
# Huang et al. 2020 *methods 1, 2, 3* relative attention (MULTIPLICATIVE).
#
# Where method 4 (above) ADDS two relative dot-product terms to the logit, Huang
# methods 1-3 instead MODULATE (multiply) the content dot-product q_i . k_j by a
# learnable relative-position term, keyed by the query->key offset d = j - i:
#
#   method 1 (eq. 11-12): e_ij = (q_i . k_j) * a_{|d|}   / sqrt(hd)
#       a is a learnable per-head SCALAR keyed by the ABSOLUTE distance |d| (no
#       sign).  The content score is uniformly scaled by a single number.
#   method 2 (eq. 13):    e_ij = (q_i . k_j) * a_{d}     / sqrt(hd)
#       same as m1 but the per-head SCALAR is keyed by the SIGNED distance d, so
#       left/right neighbours can be treated asymmetrically.
#   method 3 (eq. 15):    e_ij = ( SUM_c q_{i,c} k_{j,c} a_{d,c} ) / sqrt(hd)
#       a is a learnable per-head VECTOR (dim = head_dim) keyed by the SIGNED
#       distance: the elementwise product ("gate") of the THREE vectors q_i, k_j
#       and a_d, summed over the head dimension.  Milder than method 4's additive
#       q.a + k.a interactions but still content x position coupled per channel.
#
# The paper states m1 uses no clipping; for a fixed-size table we reuse the SAME
# clip window k = max_distance as method 4 (offsets beyond +/-k share the boundary
# entry), noted in the class.  The signed clipped offset id = clip(j-i)+k in
# [0, 2k] is the very rel_index the method-4 encoder already builds, so all four
# methods share the encoder's internal clip infrastructure unchanged.
#
# INIT: every table is initialised to 1.0.  Crucially the tables are raw
# ``nn.Parameter`` (NOT ``nn.Embedding``), so ``MusicJepa._init_weights`` -- which
# resets ``nn.Embedding`` weights to ``normal(0, init_std)`` -- does NOT touch
# them.  With a == 1 we get e_ij == (q_i . k_j)/sqrt(hd) exactly, i.e. the module
# starts at STANDARD scaled dot-product attention and learns a departure from it.
# (Method 4 wants the opposite -- a ~= 0 at init -- so it deliberately keeps its
# table as an ``nn.Embedding`` that _init_weights shrinks to ~0.)
# ---------------------------------------------------------------------------
class RelativeScaleMultiheadAttention(nn.Module):
    """Multi-head self-attention with Huang-2020 *method 1/2/3* relative modulation.

    Same projections / masking / ``batch_first`` layout and the same shared
    clipped ``rel_index`` input as :class:`RelativeVectorMultiheadAttention`, but
    the relative term MULTIPLIES the content dot-product instead of being added.
    ``method in {1, 2, 3}`` selects the exact formula (see module comment).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float,
                 max_distance: int = 128, method: int = 2):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert method in (1, 2, 3), "RelativeScaleMultiheadAttention handles methods 1/2/3"
        self.method = int(method)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_distance = max_distance
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        k = max_distance
        if self.method == 1:
            # scalar per ABSOLUTE distance |d| in [0, k], per head.
            self.rel_scale = nn.Parameter(torch.ones(k + 1, num_heads))
        elif self.method == 2:
            # scalar per SIGNED clipped distance id in [0, 2k], per head.
            self.rel_scale = nn.Parameter(torch.ones(2 * k + 1, num_heads))
        else:  # method 3
            # VECTOR (dim=head_dim) per SIGNED clipped distance id, per head.
            self.rel_scale = nn.Parameter(torch.ones(2 * k + 1, num_heads, self.head_dim))

    def forward(self, x, rel_index, key_padding_mask=None):
        b, s, _ = x.shape
        h, hd = self.num_heads, self.head_dim

        def split(t):
            return t.view(b, s, h, hd).transpose(1, 2)              # [B,H,S,hd]

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))

        if self.method in (1, 2):
            if self.method == 1:
                ids = (rel_index - self.max_distance).abs()         # |d| in [0,k]
            else:
                ids = rel_index                                     # d+k in [0,2k]
            a = self.rel_scale[ids].to(q.dtype)                     # [S,S,H]
            content = torch.matmul(q, k.transpose(-2, -1))          # [B,H,S,S] q.k
            scores = content * a.permute(2, 0, 1).unsqueeze(0)      # * a_{d}
            scores = scores * self.scale                            # /sqrt(hd)
        else:  # method 3: per-head vector gate  sum_c q_ic k_jc a_{d,c}
            a_pair = self.rel_scale[rel_index].to(q.dtype)          # [S,S,H,hd]
            scores = torch.einsum("bhic,ijhc,bhjc->bhij", q, a_pair, k)
            scores = scores * self.scale                            # /sqrt(hd)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)                                # [B,H,S,hd]
        out = out.transpose(1, 2).reshape(b, s, h * hd)
        return self.out_proj(out)


class RelativeScaleEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer using :class:`RelativeScaleMultiheadAttention`.

    Identical wiring to :class:`RelativeVectorEncoderLayer` (method 4) apart from
    the attention module, so methods 1/2/3 differ from method 4 only in the
    relative term of the attention scores.  Consumes the same precomputed
    clipped ``rel_index`` matrix.
    """

    def __init__(self, d_model, num_heads, ffn, dropout, max_distance=128, method=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelativeScaleMultiheadAttention(
            d_model, num_heads, dropout, max_distance, method)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ffn)
        self.activation = nn.GELU()
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, rel_index, key_padding_mask=None):
        x = x + self.dropout1(self.attn(self.norm1(x), rel_index, key_padding_mask))
        h = self.linear2(self.dropout_ffn(self.activation(self.linear1(self.norm2(x)))))
        x = x + self.dropout2(h)
        return x


class RelativeTransformerEncoder(nn.Module):
    """Stack of relative-attention encoder layers + final LayerNorm.

    Drop-in replacement for the ``nn.TransformerEncoder`` produced by
    :func:`_make_pre_ln_encoder`: same ``forward(x, src_key_padding_mask=...)``
    signature and the same trailing LayerNorm (pre-LN convention).

    ``rel_attn_method`` selects the Huang et al. 2020 relative-attention flavour:
    1/2/3 = MULTIPLICATIVE modulation (:class:`RelativeScaleEncoderLayer`),
    4 = ADDITIVE relative vectors (:class:`RelativeVectorEncoderLayer`, eq. 16).
    All four build the same internal clipped ``rel_index`` per sequence, so
    downstream eval is method-agnostic.
    """

    def __init__(self, d_model, num_heads, ffn, dropout, num_layers,
                 num_buckets=32, max_distance=128, rel_attn_method=2,
                 final_norm=True):
        super().__init__()
        if rel_attn_method not in (1, 2, 3, 4):
            raise ValueError(
                "rel_attn_method must be 1, 2, 3 or 4, got {!r}".format(rel_attn_method))
        self.rel_attn_method = int(rel_attn_method)
        self.max_distance = max_distance
        if self.rel_attn_method == 4:
            # Huang method 4: per-layer learnable relative VECTORS added to the
            # logit (each attention owns its own nn.Embedding table).
            self.layers = nn.ModuleList([
                RelativeVectorEncoderLayer(d_model, num_heads, ffn, dropout, max_distance)
                for _ in range(num_layers)
            ])
        else:
            # Huang methods 1/2/3: per-layer learnable MULTIPLICATIVE relative
            # tables (scalar for 1/2, per-head vector for 3), each owned by its
            # attention module and initialised to 1.0 (identity at init).
            self.layers = nn.ModuleList([
                RelativeScaleEncoderLayer(
                    d_model, num_heads, ffn, dropout, max_distance, self.rel_attn_method)
                for _ in range(num_layers)
            ])
        self.norm = nn.LayerNorm(d_model) if final_norm else nn.Identity()

    def forward(self, x, src_key_padding_mask=None, return_layers: bool = False):
        s = x.shape[1]
        # Shared clipped signed relative-index matrix [S,S] in [0, 2k]; every
        # method derives its per-pair table entry from this (methods 1/2/3 via a
        # multiply, method 4 via the two added dot-products).
        rel_index = _relative_clip_index(s, s, self.max_distance, x.device)  # [S,S] long
        if return_layers:
            # Per-layer outputs (each through the shared final norm) for the
            # layer-weighted probe (YAML: model.probe_layer_weights, ELMo/SUPERB-style).
            outs = []
            for layer in self.layers:
                x = layer(x, rel_index, src_key_padding_mask)
                outs.append(self.norm(x))
            return outs
        for layer in self.layers:
            x = layer(x, rel_index, src_key_padding_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Input embedding (the paper's Emb layer).
# ---------------------------------------------------------------------------
class JepaInputEmbedding(nn.Module):
    """``Emb(x) = W_E[Oct(x); FME_pitch(x); FME_dur(x)] + PE(x)`` with [MASK] subst.

    Input ``tokens`` are OctupleMIDI **field values** ``[B, S, 8]`` (as produced
    by :mod:`src.data.jepa_datamodule`), NOT global vocab ids.

    The absolute index-based ``PE(x)`` term is controlled by ``cfg.pos_encoding``
    (see :class:`MusicJepaConfig`): it is added in ``'absolute'`` (baseline /
    default) and ``'absolute_relative'`` (the paper main model) mode.  In
    ``'none'`` and ``'relative'`` mode it is omitted; those two (plus
    ``'absolute_relative'``) instead rely on the relative attention inside
    :class:`RelativeTransformerEncoder` (built by :class:`JepaEncoder`).
    """

    def __init__(self, cfg: MusicJepaConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.pos_encoding not in POS_ENCODING_MODES:
            raise ValueError("pos_encoding must be one of {}, got {!r}".format(
                POS_ENCODING_MODES, cfg.pos_encoding))
        self.pos_encoding = cfg.pos_encoding
        d = cfg.d_model
        # 8 per-field embedding tables (paper's E_1..E_8), sized to field-value space.
        self.field_embeddings = nn.ModuleList(
            [nn.Embedding(cfg.field_sizes[j], cfg.field_embed_dim) for j in range(NUM_FIELDS)]
        )
        # W_O: concat of 8 field embeddings -> Oct(x)  (paper: Oct = W_O[E1;...;E8]).
        self.w_o = nn.Linear(NUM_FIELDS * cfg.field_embed_dim, d)
        # Fundamental Music Embedding on pitch + duration fields.
        self.octuple_fme = OctupleFME(
            d_fme=cfg.d_fme, pitch_base=cfg.pitch_base, duration_base=cfg.duration_base,
            if_trainable=cfg.fme_trainable, translation_bias=cfg.fme_translation_bias,
            duration_unit=cfg.duration_unit,
        )
        # W_E: [Oct ; FME_pitch ; FME_dur] -> d_model  (paper's Emb projection).
        # use_fme=False ablates the FME term: W_E projects Oct alone.
        self.use_fme = bool(getattr(cfg, "use_fme", True))
        self.w_e = nn.Linear(d + (2 * cfg.d_fme if self.use_fme else 0), d)
        # Learned [MASK] embedding substituted at hidden context positions.
        self.mask_token = nn.Parameter(torch.zeros(d))
        # Absolute sinusoidal PE (fixed buffer).
        self.register_buffer("pe", build_sinusoidal_pe(cfg.max_positions, d), persistent=False)
        self.dropout = nn.Dropout(cfg.encoder_dropout)
        nn.init.normal_(self.mask_token, std=cfg.init_std)

    def forward(
        self,
        tokens: torch.Tensor,                 # [B, S, 8] field values (long)
        pad_mask: Optional[torch.Tensor] = None,      # [B, S] True = real note
        visible_mask: Optional[torch.Tensor] = None,  # [B, S] True = visible ctx
    ) -> torch.Tensor:
        b, s, nf = tokens.shape
        assert nf == NUM_FIELDS, "expected 8 octuple fields, got {}".format(nf)
        # Oct(x) = W_O[E1;...;E8]
        field_embs = [self.field_embeddings[j](tokens[..., j]) for j in range(NUM_FIELDS)]
        oct_cat = torch.cat(field_embs, dim=-1)          # [B, S, 8*fe]
        oct = self.w_o(oct_cat)                          # [B, S, d]
        # FME fusion (ablatable)
        if self.use_fme:
            fme = self.octuple_fme(tokens)               # [B, S, 2*d_fme]
            content = self.w_e(torch.cat([oct, fme], dim=-1))  # [B, S, d]
        else:
            content = self.w_e(oct)                       # [B, S, d]
        # [MASK] substitution at hidden (real) positions
        if visible_mask is not None:
            sub = ~visible_mask
            if pad_mask is not None:
                sub = sub & pad_mask
            content = torch.where(sub.unsqueeze(-1), self.mask_token.to(content.dtype), content)
        # + absolute PE ('absolute' baseline + 'absolute_relative' main model;
        # 'none'/'relative' omit the index-based PE)
        if self.pos_encoding in _ABS_PE_MODES:
            content = content + self.pe[:s].unsqueeze(0).to(content.dtype)
        return self.dropout(content)


# ---------------------------------------------------------------------------
# Context / target encoder.
# ---------------------------------------------------------------------------
class JepaEncoder(nn.Module):
    """Transformer encoder ``f_theta`` over the JEPA input embedding.

    ``forward`` returns per-position hidden states ``[B, S, d_model]``.  Pass
    ``visible_mask`` (= ``context_mask``) for the context branch to activate the
    [MASK] substitution; pass ``None`` for the target branch (clean full view).
    """

    def __init__(self, cfg: MusicJepaConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = JepaInputEmbedding(cfg)
        if cfg.pos_encoding in _REL_ATTN_MODES:
            # Relative key-query attention encoder.  'relative' = paper ablation
            # (no absolute PE); 'absolute_relative' = paper main model (absolute PE
            # added in the embedding above, relative attention here).  rel_attn_method
            # selects scalar-bias (2, default) vs vector method-4 (4).
            self.transformer = RelativeTransformerEncoder(
                cfg.d_model, cfg.encoder_heads, cfg.encoder_ffn,
                cfg.encoder_dropout, cfg.encoder_layers,
                num_buckets=cfg.rel_pos_num_buckets,
                max_distance=cfg.rel_pos_max_distance,
                rel_attn_method=getattr(cfg, "rel_attn_method", 2),
                final_norm=getattr(cfg, "encoder_final_norm", True),
            )
        else:
            self.transformer = _make_pre_ln_encoder(
                cfg.d_model, cfg.encoder_heads, cfg.encoder_ffn,
                cfg.encoder_dropout, cfg.encoder_layers,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        visible_mask: Optional[torch.Tensor] = None,
        return_layers: bool = False,
    ) -> torch.Tensor:
        x = self.embedding(tokens, pad_mask=pad_mask, visible_mask=visible_mask)
        key_padding_mask = (~pad_mask) if pad_mask is not None else None
        if return_layers:
            if not isinstance(self.transformer, RelativeTransformerEncoder):
                raise ValueError("return_layers requires the relative-attention encoder")
            return self.transformer(x, src_key_padding_mask=key_padding_mask,
                                    return_layers=True)
        return self.transformer(x, src_key_padding_mask=key_padding_mask)


# ---------------------------------------------------------------------------
# Predictor.
# ---------------------------------------------------------------------------
class JepaPredictor(nn.Module):
    """I-JEPA-style predictor with masking-latent conditioning.

    Inputs: context representations ``[B, S, d_model]`` (from ``f_theta``), their
    padding mask, the ``T`` absolute target positions to predict, and the latent
    variable (mask type id + numeric params).  Output: ``yhat [B, T, d_model]``
    (predicted target representations, one per target position).
    """

    def __init__(self, cfg: MusicJepaConfig):
        super().__init__()
        self.cfg = cfg
        dp = cfg.predictor_dim
        self.proj_in = nn.Linear(cfg.d_model, dp)
        self.register_buffer("pe", build_sinusoidal_pe(cfg.max_positions, dp), persistent=False)
        self.query_token = nn.Parameter(torch.zeros(dp))
        self.mask_type_embedding = nn.Embedding(cfg.num_mask_types, dp)
        self.params_mlp = nn.Sequential(
            nn.Linear(cfg.latent_param_dim, dp), nn.GELU(), nn.Linear(dp, dp),
        )
        self.transformer = _make_pre_ln_encoder(
            dp, cfg.predictor_heads, cfg.predictor_ffn,
            cfg.predictor_dropout, cfg.predictor_layers,
        )
        self.proj_out = nn.Linear(dp, cfg.d_model)
        nn.init.normal_(self.query_token, std=cfg.init_std)

    def forward(
        self,
        context: torch.Tensor,               # [B, S, d_model]
        context_pad_mask: Optional[torch.Tensor],  # [B, S] True = real
        target_positions: torch.Tensor,      # [B, T] long, -1 = padded slot
        latent_mask_type: torch.Tensor,      # [B] long
        latent_params: torch.Tensor,         # [B, P] float
    ) -> torch.Tensor:
        b, s, _ = context.shape
        t = target_positions.shape[1]
        dp = self.cfg.predictor_dim

        ctx = self.proj_in(context) + self.pe[:s].unsqueeze(0).to(context.dtype)  # [B,S,dp]

        # latent conditioning vector, added to each query token
        latent_vec = self.mask_type_embedding(latent_mask_type) \
            + self.params_mlp(latent_params.to(ctx.dtype))                        # [B,dp]

        valid_q = target_positions >= 0                                          # [B,T]
        tpos = target_positions.clamp(min=0)                                     # [B,T]
        q_pe = self.pe.to(ctx.dtype)[tpos]                                       # [B,T,dp]
        q = self.query_token.view(1, 1, dp).to(ctx.dtype) + q_pe + latent_vec.unsqueeze(1)

        seq = torch.cat([ctx, q], dim=1)                                         # [B,S+T,dp]
        ctx_pad = (~context_pad_mask) if context_pad_mask is not None \
            else torch.zeros(b, s, dtype=torch.bool, device=seq.device)
        key_padding_mask = torch.cat([ctx_pad, ~valid_q], dim=1)                 # True = pad
        seq = self.transformer(seq, src_key_padding_mask=key_padding_mask)
        q_out = seq[:, s:, :]                                                    # [B,T,dp]
        return self.proj_out(q_out)                                             # [B,T,d_model]


# ---------------------------------------------------------------------------
# Full model.
# ---------------------------------------------------------------------------
class MusicJepa(nn.Module):
    """Context encoder + EMA target encoder + latent-conditioned predictor.

    ``forward(batch)`` returns ``(yhat, y, valid)`` where ``yhat``/``y`` are
    ``[B, T, d_model]`` (predicted / target representations gathered at the target
    positions) and ``valid`` is ``[B, T]`` bool marking real (non-padded) target
    slots.  Loss/weighting lives in :mod:`src.utils.vicreg` and the LightningModule.
    """

    def __init__(self, cfg: Optional[MusicJepaConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else MusicJepaConfig()
        self.context_encoder = JepaEncoder(self.cfg)
        self.predictor = JepaPredictor(self.cfg)
        self.apply(self._init_weights)
        # Target encoder: deepcopy AFTER init so theta_EMA(0) == theta(0); frozen + eval.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.target_encoder.eval()

    # -- initialization (BERT-style; only used when training from scratch) -----
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cfg.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # keep target encoder in eval mode even when the module is set to train()
    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def copy_target_from_context(self) -> None:
        """Hard-reset the target encoder to the current context encoder weights."""
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def encode_context(self, batch) -> torch.Tensor:
        return self.context_encoder(
            batch["context_tokens"], pad_mask=batch["pad_mask"],
            visible_mask=batch["context_mask"],
        )

    @torch.no_grad()
    def encode_target(self, batch) -> torch.Tensor:
        self.target_encoder.eval()
        return self.target_encoder(
            batch["target_tokens"], pad_mask=batch["pad_mask"], visible_mask=None,
        )

    def forward(self, batch, return_context: bool = False,
                return_target: bool = False):
        # ``return_target`` exposes the EMA target encoding, already computed below,
        # so an equivariance term can use it as the unshifted view z1 without a
        # second forward. Same argument as ``return_context``: it is free.
        context = self.encode_context(batch)                 # [B,S,d]
        target = self.encode_target(batch)                   # [B,S,d] (no grad)

        target_positions = batch["target_positions"]         # [B,T]
        valid = target_positions >= 0
        d = target.shape[-1]
        idx = target_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, d)
        y = target.gather(1, idx)                            # [B,T,d]

        yhat = self.predictor(
            context, batch["pad_mask"], target_positions,
            batch["latent_mask_type"], batch["latent_params"],
        )                                                    # [B,T,d]
        # ``return_context`` (default False -> unchanged 3-tuple) additionally
        # exposes the *context representation* so the training loop can regularize
        # / monitor the tensor that downstream probing actually consumes (the
        # paper's VICReg only touches ``yhat``, which is decoupled from it and
        # cannot prevent a constant-collapse of the encoder).  ``context`` is
        # already computed above, so this is free.
        if return_context and return_target:
            return yhat, y, valid, context, target
        if return_context:
            return yhat, y, valid, context
        if return_target:
            return yhat, y, valid, target
        return yhat, y, valid

    # -- convenience ---------------------------------------------------------
    @staticmethod
    def flatten_valid(yhat, y, valid):
        """Select valid target rows -> ``(yhat[M,d], y[M,d])``."""
        return yhat[valid], y[valid]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_music_jepa(**overrides) -> MusicJepa:
    return MusicJepa(MusicJepaConfig(**overrides))
