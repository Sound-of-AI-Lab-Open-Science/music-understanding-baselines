"""MusicBERT backbone wearing the JEPA encoder's interface.

Purpose
-------
Expose the *pretrained MusicBERT* representation through the same
``forward(octuple, pad_mask) -> [B, N, D]`` signature
:class:`src.models.jepa.JepaEncoder` offers, so an evaluation can hold
everything else constant and vary only the encoder that produced the per-note
hidden states.  This is what the evaluation library's MusicBERT worker
(``ours_workers/musicbert_worker.py``) builds on.

Why an adapter is needed
------------------------
The two encoders speak different input languages:

* :class:`src.models.jepa.JepaEncoder` consumes ``octuple [B, N, 8]`` of
  **per-field values** (field ``j`` in ``[0, FIELD_SIZE[j])``) and returns
  ``[B, N, d_model]`` -- one vector per NOTE.
* :class:`src.models.musicbert.MusicBert` consumes a **flat global-vocab id**
  stream ``[B, 8*S]`` (8 element-tokens per note, fairseq specials
  ``bos=0 pad=1 eos=2 unk=3``) and -- via its ``embeddings.downsampling``
  ``Linear(8*768 -> 768)`` -- also produces one vector per note at octuple
  resolution, ``[B, S, 768]``.

So the conversion is purely on the input side: ``global_id[:, :, j] =
octuple[:, :, j] + FIELD_BASE_ID[j]``, flatten to ``[B, 8*N]``, and overwrite
padded notes with MusicBERT's ``pad_token_id`` (1) in all 8 slots so that
``MusicBert._padding_mask`` / ``_make_positions`` see them as padding.

Fail-loudly weight loading
--------------------------
Loading is ``strict=True`` **after** an explicit key-set diff, and additionally
asserts that the full backbone arrived: the embedding block plus one complete
set of tensors for every one of the 12 encoder layers.  Anything short of that
raises, so a key mismatch can never be swallowed and leave a randomly
initialised encoder scoring as MusicBERT.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.masking import FIELD_BASE_ID, FIELD_SIZE, NUM_FIELDS  # noqa: E402
from src.models.musicbert import MusicBert, MusicBertConfig  # noqa: E402


class MusicBertProbeAdapter(nn.Module):
    """Pretrained MusicBERT exposed as ``forward(octuple, pad_mask) -> [B, N, D]``.

    Parameters
    ----------
    init_from:
        Path to ``checkpoints/musicbert_base_converted.pt`` (a
        ``{"config": ..., "state_dict": ...}`` blob written by
        :meth:`src.models.musicbert.MusicBert.save_pretrained`).
    prepend_bos:
        Prepend one ``<s>`` octuple (8 x id 0) before encoding and drop its
        hidden state afterwards, so the note sequence sits at the same absolute
        positions MusicBERT saw during pre-training (fairseq always feeds
        ``<s> ... </s>``).  The returned tensor is ``[B, N, D]`` either way -- the
        probe head never sees the extra step.  Default True (in-distribution for
        MusicBERT); set ``model.mb_prepend_bos: false`` to feed the raw note
        stream, which is byte-for-byte the same token content our JEPA encoder
        gets.
    drop_lm_head:
        Delete ``lm_head`` / ``upsampling`` *after* the strict load (they are
        verified to be present, then discarded).  They are unused by the probe
        and would otherwise add ~13.6M dead parameters to every saved
        checkpoint.  The strictness of the load is unaffected.
    """

    #: tensors that must be present for the backbone to count as "fully loaded"
    _EMBEDDING_KEYS = (
        "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "embeddings.downsampling.weight",
        "embeddings.downsampling.bias",
        "embeddings.layer_norm.weight",
        "embeddings.layer_norm.bias",
    )
    #: per-layer tensor suffixes (16 per post-LN MusicBERT layer)
    _LAYER_SUFFIXES = (
        "self_attn.q_proj.weight", "self_attn.q_proj.bias",
        "self_attn.k_proj.weight", "self_attn.k_proj.bias",
        "self_attn.v_proj.weight", "self_attn.v_proj.bias",
        "self_attn.out_proj.weight", "self_attn.out_proj.bias",
        "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
        "fc1.weight", "fc1.bias",
        "fc2.weight", "fc2.bias",
        "final_layer_norm.weight", "final_layer_norm.bias",
    )

    def __init__(self, init_from: str, *, prepend_bos: bool = True,
                 drop_lm_head: bool = True):
        super().__init__()
        if not init_from:
            raise ValueError("MusicBertProbeAdapter requires init_from "
                             "(the converted MusicBERT checkpoint)")
        if not os.path.exists(init_from):
            raise FileNotFoundError(
                "MusicBERT checkpoint not found: {}".format(init_from))

        blob = torch.load(init_from, map_location="cpu", weights_only=False)
        if not (isinstance(blob, dict) and "state_dict" in blob and "config" in blob):
            raise RuntimeError(
                "{} is not a converted MusicBERT blob: expected a dict with "
                "'config' and 'state_dict' keys, got {}".format(
                    init_from,
                    sorted(blob.keys())[:8] if isinstance(blob, dict) else type(blob)))
        state = blob["state_dict"]
        config = MusicBertConfig.from_dict(blob["config"])
        self.config = config
        self.mb = MusicBert(config)

        n_loaded = self._strict_load(state, init_from)

        if drop_lm_head:
            # verified present by _strict_load above; unused by the probe
            del self.mb.lm_head
            del self.mb.upsampling

        self.prepend_bos = bool(prepend_bos)
        self.pad_token_id = int(config.pad_token_id)
        self.bos_token_id = int(config.bos_token_id)
        self.vocab_size = int(config.vocab_size)
        self.register_buffer(
            "field_base",
            torch.tensor(FIELD_BASE_ID, dtype=torch.long).view(1, 1, NUM_FIELDS),
            persistent=False)
        self.register_buffer(
            "field_size",
            torch.tensor(FIELD_SIZE, dtype=torch.long).view(1, 1, NUM_FIELDS),
            persistent=False)
        self._ids_checked = False

        n_params = sum(p.numel() for p in self.mb.parameters())
        print("[musicbert-adapter] loaded {} tensors STRICTLY from {} "
              "({} layers, d_model={}, {:.1f}M params kept, prepend_bos={})".format(
                  n_loaded, init_from, config.num_hidden_layers,
                  config.hidden_size, n_params / 1e6, self.prepend_bos),
              flush=True)

    # -- interface the probe expects ----------------------------------------
    @property
    def d_model(self) -> int:
        return int(self.config.hidden_size)

    # -- weight loading ------------------------------------------------------
    def _strict_load(self, state: dict, path: str) -> int:
        """Load ``state`` with a hard key-set check; return #tensors loaded."""
        expected = set(self.mb.state_dict().keys())
        got = set(state.keys())
        missing = sorted(expected - got)
        unexpected = sorted(got - expected)
        if missing or unexpected:
            raise RuntimeError(
                "MusicBERT checkpoint {} does not match the MusicBert module: "
                "{} missing (e.g. {}), {} unexpected (e.g. {}). Refusing to run "
                "with a partially initialised backbone.".format(
                    path, len(missing), missing[:5], len(unexpected), unexpected[:5]))
        self.mb.load_state_dict(state, strict=True)

        # Belt and braces: the key-set diff above already guarantees this, but
        # spell the requirement out so a future MusicBert refactor that renames
        # or drops layers cannot quietly shrink the backbone.
        for k in self._EMBEDDING_KEYS:
            if k not in state:
                raise RuntimeError("MusicBERT checkpoint is missing {}".format(k))
        for i in range(int(self.config.num_hidden_layers)):
            for suf in self._LAYER_SUFFIXES:
                k = "encoder.layers.{}.{}".format(i, suf)
                if k not in state:
                    raise RuntimeError(
                        "MusicBERT checkpoint is missing encoder layer tensor "
                        "{} -- backbone is not fully loaded".format(k))
        n_backbone = sum(1 for k in state
                         if k.startswith("embeddings.") or k.startswith("encoder."))
        exp_backbone = len(self._EMBEDDING_KEYS) + \
            int(self.config.num_hidden_layers) * len(self._LAYER_SUFFIXES)
        if n_backbone != exp_backbone:
            raise RuntimeError(
                "MusicBERT backbone tensor count {} != expected {}".format(
                    n_backbone, exp_backbone))
        # A checkpoint of the right shape but all-zero would still be "loaded";
        # a non-trivial norm on the deepest layer proves real weights arrived.
        top = state["encoder.layers.{}.fc2.weight".format(
            int(self.config.num_hidden_layers) - 1)]
        if not torch.isfinite(top).all() or float(top.float().abs().sum()) == 0.0:
            raise RuntimeError(
                "MusicBERT top-layer weights are zero/non-finite -- refusing to "
                "report this as a pretrained representation")
        return len(state)

    # -- id conversion -------------------------------------------------------
    def octuple_to_global_ids(self, octuple: torch.Tensor,
                              pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """``[B, N, 8]`` field values -> ``[B, N, 8]`` MusicBERT global vocab ids.

        Padded notes (``pad_mask == False``; the probe collate uses True = VALID)
        become ``pad_token_id`` in all 8 slots, which is what
        ``MusicBert._padding_mask`` and fairseq ``make_positions`` key off.
        """
        if octuple.dim() != 3 or octuple.shape[-1] != NUM_FIELDS:
            raise ValueError("expected octuple [B, N, 8], got {}".format(
                tuple(octuple.shape)))
        ids = octuple.long() + self.field_base
        if pad_mask is not None:
            if pad_mask.shape != octuple.shape[:2]:
                raise ValueError("pad_mask {} does not match octuple {}".format(
                    tuple(pad_mask.shape), tuple(octuple.shape[:2])))
            ids = torch.where(pad_mask.bool().unsqueeze(-1), ids,
                              torch.full_like(ids, self.pad_token_id))
        return ids

    def _check_ids_once(self, octuple: torch.Tensor, ids: torch.Tensor,
                        pad_mask: Optional[torch.Tensor]) -> None:
        """One-off (first batch) range validation; costs one host sync, once."""
        if self._ids_checked:
            return
        self._ids_checked = True
        valid = pad_mask.bool() if pad_mask is not None else torch.ones(
            octuple.shape[:2], dtype=torch.bool, device=octuple.device)
        if not bool(valid.any()):
            return
        v = octuple.long()[valid]                       # [M, 8] real notes only
        lo = int(v.min()) if v.numel() else 0
        if lo < 0:
            raise RuntimeError("octuple field value {} < 0".format(lo))
        over = (v >= self.field_size.view(1, NUM_FIELDS))
        if bool(over.any()):
            j = int(over.any(dim=0).nonzero()[0])
            raise RuntimeError(
                "octuple field {} exceeds FIELD_SIZE[{}]={} (max seen {})".format(
                    j, j, FIELD_SIZE[j], int(v[:, j].max())))
        g = ids.long()[valid]
        gmin, gmax = int(g.min()), int(g.max())
        if gmin < 0 or gmax >= self.vocab_size:
            raise RuntimeError(
                "global ids out of vocab range: [{}, {}] vs vocab_size {}".format(
                    gmin, gmax, self.vocab_size))
        print("[musicbert-adapter] id check OK: {} real notes, global ids in "
              "[{}, {}] (vocab {})".format(int(valid.sum()), gmin, gmax,
                                           self.vocab_size), flush=True)

    # -- forward -------------------------------------------------------------
    def forward(self, octuple: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None,
                visible_mask: Optional[torch.Tensor] = None,
                return_layers: bool = False) -> torch.Tensor:
        """``octuple [B, N, 8]`` field values -> ``[B, N, hidden_size]``.

        ``pad_mask [B, N]`` follows the probe convention: **True = valid note**.
        ``visible_mask`` (the JEPA context branch's [MASK] substitution) has no
        MusicBERT equivalent and is rejected rather than silently ignored.
        """
        if visible_mask is not None:
            raise NotImplementedError(
                "MusicBertProbeAdapter has no [MASK]-substitution path; the probe "
                "must pass visible_mask=None (clean full view)")
        if return_layers:
            raise NotImplementedError(
                "MusicBertProbeAdapter does not expose per-layer hidden states "
                "(model.probe_layer_weights is unsupported for this backbone)")

        b, n, _ = octuple.shape
        ids = self.octuple_to_global_ids(octuple, pad_mask)      # [B, N, 8]
        self._check_ids_once(octuple, ids, pad_mask)
        if self.prepend_bos:
            bos = ids.new_full((b, 1, NUM_FIELDS), self.bos_token_id)
            ids = torch.cat([bos, ids], dim=1)                   # [B, N+1, 8]
        flat = ids.reshape(b, -1)                                # [B, 8*(N[+1])]
        hidden = self.mb.encode(flat, upsample=False)            # [B, N[+1], D]
        if self.prepend_bos:
            hidden = hidden[:, 1:, :]
        if hidden.shape[:2] != (b, n):
            raise RuntimeError("adapter produced {} notes, expected {}".format(
                tuple(hidden.shape[:2]), (b, n)))
        return hidden
