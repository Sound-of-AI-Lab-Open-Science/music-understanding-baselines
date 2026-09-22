"""Swin Transformer V2 encoder/decoder/MEP-head for midi-rae.

Ported from upstream ``midi_rae/swin.py`` (see package __init__), TRIMMED to
what ``model.encoder: swin`` with ``training.lambda_mae: 0.0`` actually uses:
``SwinEncoder``, ``SwinDecoder`` (for the frozen-encoder decoder stage) and
``SwinMaskedEmbeddingPredictor`` (the MEP loss, ``lambda_mep`` > 0 in both
configs). ``SwinMAEDecoder``/``PixelShuffleHead`` are dropped along with them
(both configs carry ``lambda_mae: 0.0``, and upstream's ``SwinMAEDecoder`` only
exists to serve that path) -- dropping them also removes the only dependency
upstream's ``swin.py`` has on ``vit.py`` (``TransformerBlock``), so nothing
from the ViT half of midi-rae needs to be vendored at all.
"""

from __future__ import annotations

from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import calculate_drop_path_rates, trunc_normal_
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from .core import EncoderOutput, HierarchicalPatchState, PatchState


class SwinEncoder(nn.Module):
    "Swin Transformer V2 Encoder for midi-rae."
    def __init__(self, img_height: int, img_width: int, patch_h: int = 4, patch_w: int = 4,
                 in_chans: int = 1, embed_dim: int = 8, depths: tuple = (1, 2, 2, 6, 2, 1),
                 num_heads: tuple = (1, 1, 2, 4, 8, 16), window_size: int = 8,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0.,
                 proj_drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer: type = nn.LayerNorm,
                 mae_ratio: float = 0., empty_mask_ratio: float = 0.05,
                 squash_coarse=False):
        super().__init__()
        self.num_stages, self.embed_dim = len(depths), embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.patch_h, self.patch_w, self.grid_size = patch_h, patch_w, (img_height // patch_h, img_width // patch_w)
        self.mae_ratio, self.empty_mask_ratio = mae_ratio, empty_mask_ratio

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        self.patch_norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.empty_input_token = nn.Parameter(torch.zeros(embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        embed_dims = [int(embed_dim * 2 ** i) for i in range(self.num_stages)]
        self.empty_tokens = None if (not squash_coarse) else nn.ParameterList(
            [nn.Parameter(torch.randn(d) * 1.0) for d in embed_dims])

        dpr = calculate_drop_path_rates(drop_path_rate, list(depths), stagewise=True)
        self.stages = nn.ModuleList()
        in_dim, scale = embed_dims[0], 1
        for i in range(self.num_stages):
            out_dim = embed_dims[i]
            self.stages.append(SwinTransformerV2Stage(
                dim=in_dim, out_dim=out_dim, depth=depths[i], num_heads=num_heads[i],
                input_resolution=(self.grid_size[0] // scale, self.grid_size[1] // scale),
                window_size=window_size, downsample=(i > 0), mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, proj_drop=proj_drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer))
            in_dim = out_dim
            if i > 0: scale *= 2

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)
        for stage in self.stages: stage._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        nod = {'empty_input_token', 'mask_token'}
        if self.empty_tokens is not None: nod |= {f'empty_tokens.{i}' for i in range(len(self.empty_tokens))}
        for n, _ in self.named_parameters():
            if any(kw in n for kw in ('cpb_mlp', 'logit_scale')): nod.add(n)
        return nod

    def _compute_non_empty(self, img):
        patches = img.unfold(2, self.patch_h, self.patch_h).unfold(3, self.patch_w, self.patch_w)
        return (patches.amax(dim=(-1, -2)) > 0.2).squeeze(1).flatten(1)

    def _make_mae_mask(self, non_empty, device, effective_ratio=None):
        B, N = non_empty.shape
        rand = torch.rand(B, N, device=device)
        ratio = effective_ratio if effective_ratio is not None else self.mae_ratio
        threshold = torch.where(non_empty.bool(),
            torch.full_like(rand, 1.0 - ratio),
            torch.full_like(rand, 1.0 - ratio * self.empty_mask_ratio))
        return rand < threshold

    def _make_grid_pos(self, h, w, device):
        return torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                                          indexing='ij'), dim=-1).reshape(-1, 2)

    def forward(self, x, mask_ratio: float = 0., mae_mask: Optional[torch.Tensor] = None) -> EncoderOutput:
        B, device = x.shape[0], x.device
        grid_h, grid_w = self.grid_size
        N_full = grid_h * grid_w

        non_empty = self._compute_non_empty(x)
        x = self.patch_embed(x)
        x = self.patch_norm(x.permute(0, 2, 3, 1).contiguous())
        B, H, W, C = x.shape

        effective_ratio = mask_ratio if mask_ratio > 0 else self.mae_ratio
        if mae_mask is None and effective_ratio > 0:
            mae_mask = self._make_mae_mask(non_empty, device, effective_ratio)
        if mae_mask is not None:
            m4d = mae_mask.view(B, H, W, 1)
            x = torch.where(m4d, x, self.mask_token.view(1, 1, 1, -1).expand_as(x))
        else:
            mae_mask = torch.ones(B, N_full, device=device, dtype=torch.bool)
        x = self.pos_drop(x)

        intermediates = []
        ne = non_empty.view(B, 1, grid_h, grid_w).float()
        ne_scales = []
        for i, stage in enumerate(self.stages):
            x = stage.downsample(x) if i == len(self.stages) - 1 else stage(x)
            Hf, Wf = x.shape[1], x.shape[2]
            while ne.shape[2] > Hf: ne = F.max_pool2d(ne, 2)
            if self.empty_tokens is not None:
                x = torch.where(ne.permute(0, 2, 3, 1) > 0, x, self.empty_tokens[i].view(1, 1, 1, -1).expand_as(x))
            intermediates.append(x)
            ne_scales.append(ne.view(B, -1))
        intermediates[-1] = self.norm(intermediates[-1])
        if self.empty_tokens is not None:
            intermediates[-1] = torch.where(ne.permute(0, 2, 3, 1) > 0, intermediates[-1],
                                            self.empty_tokens[-1].view(1, 1, 1, -1).expand_as(intermediates[-1]))

        levels = []
        for feat, ne_s in zip(reversed(intermediates), reversed(ne_scales)):
            Bf, Hf, Wf, Cf = feat.shape
            n = Hf * Wf
            levels.append(PatchState(
                emb=feat.reshape(Bf, n, Cf), pos=self._make_grid_pos(Hf, Wf, device),
                non_empty=ne_s,
                mae_mask=torch.ones(n, device=device, dtype=torch.bool)))

        return EncoderOutput(patches=HierarchicalPatchState(levels=levels),
            full_pos=self._make_grid_pos(grid_h, grid_w, device),
            full_non_empty=non_empty, mae_mask=mae_mask)


class PatchExpand(nn.Module):
    """Inverse of patch merging: doubles spatial resolution via learned linear expansion."""
    def __init__(self, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(in_dim, 4 * out_dim, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, 2, 2, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, 2 * H, 2 * W, -1)
        return self.norm(x)


class SwinDecoder(nn.Module):
    """Swin V2 Decoder for midi-rae -- symmetric multi-stage decoder."""
    def __init__(self, img_height: int = 128, img_width: int = 128, patch_h: int = 4, patch_w: int = 4,
                 out_channels: int = 1, embed_dim: int = 8, depths: tuple = (1, 2, 2, 6, 2, 1),
                 num_heads: tuple = (1, 1, 2, 4, 8, 16), window_size: int = 8, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop_path_rate: float = 0.1, proj_drop_rate: float = 0.,
                 attn_drop_rate: float = 0., norm_layer: type = nn.LayerNorm):
        super().__init__()
        self.patch_h, self.patch_w, self.out_channels = patch_h, patch_w, out_channels
        num_stages = len(depths)

        dec_dims = [int(embed_dim * 2 ** (num_stages - 1 - i)) for i in range(num_stages)]
        dec_depths = list(reversed(depths))
        dec_heads = list(reversed(num_heads))

        finest_h, finest_w = img_height // patch_h, img_width // patch_w
        grids = [(finest_h // 2 ** (num_stages - 1 - i), finest_w // 2 ** (num_stages - 1 - i))
                 for i in range(num_stages)]
        self.finest_h, self.finest_w = finest_h, finest_w

        self.laterals = nn.ModuleList([nn.Linear(d, d) for d in dec_dims])

        self.stages = nn.ModuleList()
        dpr = calculate_drop_path_rates(drop_path_rate, dec_depths, stagewise=True)
        for i in range(num_stages):
            gh, gw = grids[i]
            if gh <= 1 and gw <= 1:
                self.stages.append(None)
            else:
                self.stages.append(SwinTransformerV2Stage(
                    dim=dec_dims[i], out_dim=dec_dims[i], depth=dec_depths[i],
                    num_heads=dec_heads[i], input_resolution=(gh, gw),
                    window_size=window_size, downsample=False, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, proj_drop=proj_drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer))

        self.upsamples = nn.ModuleList([
            PatchExpand(dec_dims[i], dec_dims[i + 1], norm_layer) for i in range(num_stages - 1)])

        self.norm = norm_layer(dec_dims[-1])
        self.head = nn.Linear(dec_dims[-1], out_channels * patch_h * patch_w)

        self.apply(self._init_weights)
        for stage in self.stages:
            if stage is not None: stage._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        nod = set()
        for n, _ in self.named_parameters():
            if any(kw in n for kw in ('cpb_mlp', 'logit_scale')): nod.add(n)
        return nod

    def forward(self, enc_out: EncoderOutput) -> torch.Tensor:
        levels = enc_out.patches.levels
        B = levels[0].emb.shape[0]

        z = self.laterals[0](levels[0].emb)
        g = int(levels[0].pos.shape[0] ** 0.5)
        z = z.view(B, g, g, -1)
        if self.stages[0] is not None: z = self.stages[0](z)

        for i in range(len(self.upsamples)):
            z = self.upsamples[i](z)
            lat = self.laterals[i + 1](levels[i + 1].emb)
            gh, gw = z.shape[1], z.shape[2]
            z = z + lat.view(B, gh, gw, -1)
            if self.stages[i + 1] is not None: z = self.stages[i + 1](z)

        z = self.norm(z)
        z = z.reshape(B, -1, z.shape[-1])
        px = self.head(z)
        px = px.reshape(B, self.finest_h, self.finest_w, self.patch_h, self.patch_w, self.out_channels)
        return px.permute(0, 5, 1, 3, 2, 4).reshape(
            B, self.out_channels, self.finest_h * self.patch_h, self.finest_w * self.patch_w)


class SwinMaskedEmbeddingPredictor(nn.Module):
    """Perceiver-style hierarchical embedding predictor (I-JEPA-inspired), used for
    the MEP loss term (``training.lambda_mep`` > 0 in both midi_rae configs)."""
    def __init__(self, dims=(256, 128, 64, 32, 16, 8), summary_dim=128,
                 n_summaries=None, mix_depth=2, heads=4, mask_ratio=0.4, mr_level_fac=1.25):
        super().__init__()
        n_levels = len(dims)
        self.n_levels, self.mask_ratio, self.mr_level_fac = n_levels, mask_ratio, mr_level_fac

        if n_summaries is None:
            n_summaries = tuple(2 ** i for i in range(n_levels))
        self.n_summaries = n_summaries

        self.mask_tokens = nn.ParameterList([nn.Parameter(torch.randn(d) * 0.02) for d in dims])

        self.kv_projs = nn.ModuleList([nn.Linear(d, summary_dim) for d in dims])
        self.summary_queries = nn.ParameterList([
            nn.Parameter(torch.randn(ns, summary_dim) * 0.02) for ns in n_summaries])
        self.summary_attn = nn.ModuleList([
            nn.MultiheadAttention(summary_dim, heads, batch_first=True) for _ in range(n_levels)])

        self.level_emb = nn.ParameterList([
            nn.Parameter(torch.randn(1, ns, summary_dim) * 0.02) for ns in n_summaries])
        self.mix = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(summary_dim, heads, dim_feedforward=summary_dim * 4,
                                       batch_first=True, norm_first=True),
            num_layers=mix_depth, enable_nested_tensor=False)

        self.pos_proj = nn.Linear(2, summary_dim)
        self.pred_level_emb = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, summary_dim) * 0.02) for _ in range(n_levels)])
        self.pred_attn = nn.ModuleList([
            nn.MultiheadAttention(summary_dim, heads, batch_first=True) for _ in range(n_levels)])
        self.out_projs = nn.ModuleList([nn.Linear(summary_dim, d) for d in dims])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def _level_mask_ratios(self):
        return [self.mask_ratio / (self.mr_level_fac ** (self.n_levels - 1 - i)) for i in range(self.n_levels)]

    def _make_level_masks(self, levels, device):
        ratios = self._level_mask_ratios()
        masks = []
        for i, lv in enumerate(levels):
            B, N = lv.emb.shape[:2]
            masks.append(torch.rand(B, N, device=device) >= ratios[i])
        return masks

    def forward(self, enc_out: EncoderOutput, mask_ratio: Optional[float] = None):
        levels = enc_out.patches.levels
        B, device = levels[0].emb.shape[0], levels[0].emb.device

        old_ratio = self.mask_ratio
        if mask_ratio is not None: self.mask_ratio = mask_ratio
        masks = self._make_level_masks(levels, device) if self.mask_ratio > 0 else [
            torch.ones(lv.emb.shape[:2], dtype=torch.bool, device=device) for lv in levels]
        self.mask_ratio = old_ratio

        summaries = []
        for i, lv in enumerate(levels):
            emb = lv.emb.clone()
            emb[~masks[i]] = self.mask_tokens[i]
            kv = self.kv_projs[i](emb)
            q = self.summary_queries[i].unsqueeze(0).expand(B, -1, -1)
            s, _ = self.summary_attn[i](q, kv, kv)
            summaries.append(s + self.level_emb[i])
        summaries = torch.cat(summaries, dim=1)
        summaries = self.mix(summaries)

        preds = []
        for i, lv in enumerate(levels):
            q = self.pos_proj(lv.pos.float())
            q = q.unsqueeze(0).expand(B, -1, -1) + self.pred_level_emb[i]
            p, _ = self.pred_attn[i](q, summaries, summaries)
            preds.append(self.out_projs[i](p))
        return preds, masks
