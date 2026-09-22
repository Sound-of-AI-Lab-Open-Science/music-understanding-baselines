"""MusicBERT model (pure PyTorch, no fairseq / no transformers dependency).

This is a faithful, self-contained re-implementation of the official MusicBERT
encoder (Zeng et al. 2021; ``musicbert/`` in
<https://github.com/microsoft/muzic>, pin ``2b87396``) whose weights load
*1:1* from the official fairseq checkpoint.  It mirrors, layer for layer, the
fairseq ``OctupleEncoder`` (a ``TransformerSentenceEncoder`` with compound
down/up-sampling) and the RoBERTa masked-LM head:

    tokens (B, 8*S)                       # 8 element-tokens per Octuple note
      -> word_embeddings                  # (B, 8*S, D)
      -> view (B, S, 8*D) -> downsampling  Linear(8*D -> D)   # compound fusion
      -> + learned position embeddings (fairseq offset, padding_idx=1)
      -> emb LayerNorm -> dropout
      -> N x post-LN Transformer layers   # (B, S, D)  [octuple resolution]
      -> upsampling Linear(D -> 8*D) -> view (B, 8*S, D)      # back to elements
      -> lm_head (dense -> gelu -> LN -> decoder)             # (B, 8*S, vocab)

The architecture / hyper-parameters were extracted by dissecting
``checkpoints/checkpoint_last_musicbert_base.pt`` and cross-checked against the
verified fairseq->HF mapping in <https://github.com/malcolmsailor/musicbert_hf>
(pin ``64a054c``):

    encoder_layers=12, encoder_embed_dim=768, encoder_ffn_embed_dim=3072,
    encoder_attention_heads=12, activation='gelu', dropout=0.1,
    attention_dropout=0.1, activation_dropout=0.0, vocab_size=1237,
    max_positions=8192 (position table has max_positions+pad+1 = 8194 rows),
    pad=1, eos=2, unk=3, untie_weights_roberta=False, post-LayerNorm layers.

Post-LN (BERT/RoBERTa) means each sublayer is ``LN(x + sublayer(x))``.

The design goal is a clean ``nn.Module`` that the rest of the project (MLM
pre-training, JEPA context/target encoders, downstream heads) can reuse without
dragging in fairseq or a specific ``transformers`` version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Element-tokens per Octuple note (Bar, Pos, Program, Pitch, Dur, Vel, TS, Tempo)
COMPOUND_RATIO = 8


@dataclass
class MusicBertConfig:
    """Structural hyper-parameters.  Defaults are the official ``base`` model."""

    vocab_size: int = 1237
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    # fairseq ``max_positions`` (in *octuples*).  The learned position table has
    # ``max_positions + pad_token_id + 1`` rows (fairseq convention).
    max_positions: int = 8192
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    unk_token_id: int = 3
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    activation_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-5  # fairseq LayerNorm default (nn.LayerNorm default)
    activation: str = "gelu"
    compound_ratio: int = COMPOUND_RATIO

    @property
    def num_position_embeddings(self) -> int:
        # fairseq LearnedPositionalEmbedding: num_embeddings = max_positions + padding_idx + 1
        return self.max_positions + self.pad_token_id + 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MusicBertConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})


# Named architecture presets (from muzic's musicbert/musicbert/__init__.py).
ARCH_PRESETS = {
    "base": dict(num_hidden_layers=12, hidden_size=768,
                 intermediate_size=3072, num_attention_heads=12),
    "large": dict(num_hidden_layers=24, hidden_size=1024,
                  intermediate_size=4096, num_attention_heads=16),
    "medium": dict(num_hidden_layers=8, hidden_size=512,
                   intermediate_size=2048, num_attention_heads=8),
    "small": dict(num_hidden_layers=4, hidden_size=512,
                  intermediate_size=2048, num_attention_heads=8),
    "mini": dict(num_hidden_layers=4, hidden_size=256,
                 intermediate_size=1024, num_attention_heads=4),
    "tiny": dict(num_hidden_layers=2, hidden_size=128,
                 intermediate_size=512, num_attention_heads=2),
}


def config_for_arch(arch: str = "base", **overrides) -> MusicBertConfig:
    if arch not in ARCH_PRESETS:
        raise ValueError("unknown arch {!r}; choose from {}".format(
            arch, sorted(ARCH_PRESETS)))
    params = dict(ARCH_PRESETS[arch])
    params.update(overrides)
    return MusicBertConfig(**params)


def _make_positions(token_slot0: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """Replicate fairseq ``utils.make_positions`` at the octuple level.

    ``token_slot0`` is ``input_ids[:, ::compound_ratio]`` (the first element of
    each octuple; equals ``padding_idx`` exactly for padded octuples).  Non-pad
    position ids are ``cumsum(non_pad)`` offset by ``padding_idx`` so the first
    real token gets ``padding_idx + 1`` (== 2 for RoBERTa).  Pad octuples get
    ``padding_idx``.
    """
    mask = token_slot0.ne(padding_idx).int()
    positions = torch.cumsum(mask, dim=1) * mask + padding_idx
    return positions.long()


class OctupleEmbedding(nn.Module):
    """Compound word+position embedding with 8->1 down-sampling."""

    def __init__(self, config: MusicBertConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.compound_ratio = config.compound_ratio
        d = config.hidden_size
        self.word_embeddings = nn.Embedding(
            config.vocab_size, d, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.num_position_embeddings, d, padding_idx=config.pad_token_id)
        self.downsampling = nn.Linear(d * config.compound_ratio, d)
        self.layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, compound_len = input_ids.shape
        ratio = self.compound_ratio
        assert compound_len % ratio == 0, (
            "sequence length {} must be a multiple of compound_ratio {}".format(
                compound_len, ratio))
        seq = compound_len // ratio

        flat = self.word_embeddings(input_ids)                # (B, 8S, D)
        x = flat.view(b, seq, ratio * flat.shape[-1])         # (B, S, 8D)
        x = self.downsampling(x)                              # (B, S, D)

        positions = _make_positions(input_ids[:, ::ratio], self.padding_idx)
        x = x + self.position_embeddings(positions)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class MultiheadSelfAttention(nn.Module):
    """Standard multi-head self-attention (fairseq-compatible: separate q/k/v)."""

    def __init__(self, config: MusicBertConfig):
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert d % self.num_heads == 0, "hidden_size must be divisible by n_heads"
        self.head_dim = d // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, d = x.shape
        h, hd = self.num_heads, self.head_dim

        def shape(t):
            return t.view(b, s, h, hd).transpose(1, 2)  # (B, H, S, hd)

        q = shape(self.q_proj(x)) * self.scaling
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, S, S)
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) True where padding -> mask those keys
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                    # (B, H, S, hd)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(out)


class MusicBertLayer(nn.Module):
    """Post-LayerNorm Transformer encoder layer (BERT/RoBERTa style)."""

    def __init__(self, config: MusicBertConfig):
        super().__init__()
        d = config.hidden_size
        self.self_attn = MultiheadSelfAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(d, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, d)
        self.final_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation_dropout = nn.Dropout(config.activation_dropout_prob)
        self.activation_fn = _get_activation(config.activation)

    def forward(self, x, key_padding_mask=None):
        # Self-attention sublayer (post-LN)
        residual = x
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = self.dropout(x)
        x = self.self_attn_layer_norm(residual + x)
        # Feed-forward sublayer (post-LN)
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.final_layer_norm(residual + x)
        return x


class TransformerEncoderStack(nn.Module):
    def __init__(self, config: MusicBertConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [MusicBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, key_padding_mask=None, return_all_hiddens=False):
        all_hiddens = [x] if return_all_hiddens else None
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
            if return_all_hiddens:
                all_hiddens.append(x)
        return x, all_hiddens


class MusicBertLMHead(nn.Module):
    """RoBERTa masked-LM head: dense -> gelu -> LayerNorm -> decoder (+bias)."""

    def __init__(self, config: MusicBertConfig):
        super().__init__()
        d = config.hidden_size
        self.dense = nn.Linear(d, d)
        self.layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(d, config.vocab_size)  # weight + bias
        self.activation_fn = _get_activation(config.activation)

    def forward(self, x):
        x = self.activation_fn(self.dense(x))
        x = self.layer_norm(x)
        return self.decoder(x)


def _get_activation(name: str):
    if name == "gelu":
        return F.gelu  # erf-based, matches fairseq utils.get_activation_fn('gelu')
    if name == "relu":
        return F.relu
    raise ValueError("unsupported activation: {}".format(name))


class MusicBert(nn.Module):
    """MusicBERT with the masked-LM head.

    ``forward`` returns a dict with ``logits`` (B, 8*S, vocab) and, if ``labels``
    are given, the masked-LM ``loss`` (ids == ``ignore_index`` are ignored).
    """

    def __init__(self, config: MusicBertConfig):
        super().__init__()
        self.config = config
        self.compound_ratio = config.compound_ratio
        self.padding_idx = config.pad_token_id
        self.embeddings = OctupleEmbedding(config)
        self.encoder = TransformerEncoderStack(config)
        self.upsampling = nn.Linear(
            config.hidden_size, config.hidden_size * config.compound_ratio)
        self.lm_head = MusicBertLMHead(config)
        self.apply(self._init_weights)

    # -- initialization (BERT-style; only matters when training from scratch) --
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # -- helpers -------------------------------------------------------------
    def _padding_mask(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        # An octuple is padding iff its first element-token is the pad id.
        pad_mask = input_ids[:, ::self.compound_ratio].eq(self.padding_idx)
        if not pad_mask.any():
            return None
        return pad_mask

    def encode(self, input_ids: torch.Tensor, upsample: bool = False,
               return_all_hiddens: bool = False):
        """Return encoder hidden states.

        With ``upsample=False`` the output is at *octuple* resolution (B, S, D)
        -- the natural representation for pooling / JEPA.  With ``upsample=True``
        it is up-sampled back to element resolution (B, 8*S, D).
        """
        pad_mask = self._padding_mask(input_ids)
        x = self.embeddings(input_ids)
        x, all_hiddens = self.encoder(
            x, key_padding_mask=pad_mask, return_all_hiddens=return_all_hiddens)
        if upsample:
            x = self._upsample(x)
        if return_all_hiddens:
            return x, all_hiddens
        return x

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        x = self.upsampling(x)                       # (B, S, 8D)
        return x.view(b, s * self.compound_ratio, d)  # (B, 8S, D)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                ignore_index: int = -100):
        """MLM forward.

        ``input_ids``: (B, 8*S) long.  ``labels``: (B, 8*S) long with
        ``ignore_index`` at non-target positions (else the gold token id).
        """
        hidden = self.encode(input_ids, upsample=True)   # (B, 8S, D)
        logits = self.lm_head(hidden)                    # (B, 8S, vocab)
        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=ignore_index,
            )
            out["loss"] = loss
        return out

    # -- (de)serialization ---------------------------------------------------
    def save_pretrained(self, path: str):
        torch.save({"config": self.config.to_dict(),
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def from_pretrained(cls, path: str, map_location="cpu",
                        strict: bool = True) -> "MusicBert":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob:
            config = MusicBertConfig.from_dict(blob.get("config", {}))
            model = cls(config)
            model.load_state_dict(blob["state_dict"], strict=strict)
        else:  # a bare state_dict
            model = cls(MusicBertConfig())
            model.load_state_dict(blob, strict=strict)
        return model


def build_musicbert(arch: str = "base", **overrides) -> MusicBert:
    return MusicBert(config_for_arch(arch, **overrides))
