"""LeJEPA (SIGReg + attraction), factorization loss, and the decoder BCE+MSE
loss -- ported verbatim from upstream ``midi_rae/losses.py`` (see package
__init__). ``calc_mae_loss`` is dropped: it only serves the MAE-decoder path,
which both configs disable (``lambda_mae: 0.0``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def safe_mean(t, dim=None):
    return t.mean(dim=dim) if t.numel() > 0 else 0.0


def SIGReg(x, global_step, num_slices=256, chunk_size=32):
    """SIGReg with the Epps-Pulley statistic. x is (N, K). Chunked for memory."""
    with torch.amp.autocast(x.device.type, enabled=False):
        x = x.float()
        device = x.device
        g = torch.Generator(device=device).manual_seed(global_step)
        A = torch.randn((x.size(1), num_slices), generator=g, device=device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-10)
        t = torch.linspace(-5, 5, 17, device=device)
        exp_f = torch.exp(-0.5 * t ** 2)
        T_total = torch.tensor(0.0, device=device)
        if chunk_size < 1: chunk_size = num_slices
        for i in range(0, num_slices, chunk_size):
            x_t = (x @ A[:, i:i + chunk_size]).unsqueeze(2) * t
            ecf = (torch.exp(1j * x_t).mean(dim=0)).abs()
            diff = (ecf - exp_f).abs().square().mul(exp_f)
            T_total = T_total + torch.trapz(diff, t, dim=1).sum()
        return T_total


def attraction_loss(z1, z2, deltas=None, alpha=1.0, **kwargs):
    "Pull similar 'views' together, with a delta-scaled margin to prevent over-collapse."
    if deltas is None: return safe_mean((z1 - z2).square())
    if deltas.dim() == 1: deltas = deltas.unsqueeze(-1)
    dist = (z1 - z2).norm(dim=-1)
    delta_diag = (deltas ** 2).sum(dim=1)
    margin = alpha * delta_diag.sqrt()
    return safe_mean((dist - margin).clamp(min=0).square())


def factorization_loss(z_anchor, z_crop1, z_crop2, targets):
    "Encourage soft factorization/decomposition in pitch vs time."
    d1 = z_crop1 - z_anchor
    d2 = z_crop2 - z_anchor
    cos = F.cosine_similarity(d1, d2, dim=-1)
    return safe_mean((cos - targets) ** 2)


def LeJEPA(z1, z2, global_step, z3=None, valids=None, target=None, deltas=None, psize=None,
          sigreg_prefac=0.5, loss_weights=None):
    "Main LeJEPA loss: SIGReg regularizer + attraction (+ factorization when z3 given)."
    lw = loss_weights or {}
    lambd = lw.get('lambd', 0.5)
    lambda_fact = lw.get('lambda_fact', 0.5)
    lambda_sim = lw.get('lambda_sim', 1.0)
    sigreg = SIGReg(z2, global_step) * sigreg_prefac
    if z3 is not None:
        sim = attraction_loss(z1, z2, deltas=deltas[:, 0] if deltas is not None else None, psize=psize)
        fact = factorization_loss(z1, z2, z3, target)
        return {'loss': (1 - lambd) * (lambda_sim * sim + lambda_fact * fact) + lambd * sigreg,
                'sim': sim.detach(), 'sigreg': sigreg.detach(), 'fact': fact.detach()}
    sim = attraction_loss(z1, z2, deltas=deltas, psize=psize)
    return {'loss': (1 - lambd) * lambda_sim * sim + lambd * sigreg, 'sim': sim.detach(), 'sigreg': sigreg.detach()}


def anchor_loss(z1, z2):
    "Anchor embeddings of empty patches to the origin."
    return safe_mean(z1.square()) + safe_mean(z2.square())


def calc_enc_loss(z1, z2, global_step, z3=None, deltas=None, target=None,
                  non_emptys=(None, None), psize=None, loss_weights=None):
    "Main loss function for the encoder (one hierarchy level)."
    lw = loss_weights or {}
    non_empty1, non_empty2, non_empty3 = non_emptys
    loss_dict = LeJEPA(z1, z2, global_step, z3=z3,
                       valids=((non_empty1 & non_empty2).view(-1).bool(),
                               (non_empty1 & non_empty3).view(-1).bool() if non_empty3 is not None else None),
                       deltas=deltas, target=target, psize=psize, loss_weights=lw)
    lambda_anchor = lw.get('lambda_anchor', 0.05)
    if lambda_anchor > 0:
        loss_dict['anchor'] = anchor_loss(z1[~non_empty1.view(-1).bool()], z2[~non_empty2.view(-1).bool()])
        loss_dict['loss'] = loss_dict['loss'] + lambda_anchor * loss_dict['anchor']
    return loss_dict


def calc_enc_loss_multiscale(z1, z2, global_step, img_size, z3=None, deltas=None, target=None,
                             non_emptys=None, loss_weights=None, debug=False):
    """Compute encoder loss at each hierarchy level of the Swin encoder (skips
    the LeJEPA term on the finest level, matching upstream)."""
    lw = loss_weights or {}
    if not isinstance(z1, list):
        d_exp = deltas.repeat_interleave(z1.shape[0] // deltas.shape[0], dim=0)
        return calc_enc_loss(z1.float(), z2.float(), global_step, deltas=d_exp.float(),
                             non_emptys=non_emptys, loss_weights=lw)
    total = {}
    n_levels = len(z1)
    level_losses = []
    z3_iter = z3 if z3 is not None else [None] * len(z1)
    for lev, (z1_l, z2_l, z3_l, ne) in enumerate(zip(z1, z2, z3_iter, non_emptys)):
        if lev < n_levels - 1:
            B, N, D = z1_l.shape
            grid = int(N ** 0.5)
            psize = (img_size // grid, img_size // grid)
            z1_flat, z2_flat = z1_l.reshape(-1, D).float(), z2_l.reshape(-1, D).float()
            z3_flat = None
            ne_flat = list(x.reshape(-1).bool() for x in (ne[0], ne[1]))
            if z3_l is not None:
                z3_flat = z3_l.reshape(-1, D).float()
                ne_flat.append(ne[2].reshape(-1).bool())
            else:
                ne_flat.append(None)
            d_expanded = deltas.repeat_interleave(N, dim=0).float() if deltas is not None else None
            t_expanded = target.repeat_interleave(N, dim=0).float() if target is not None else None
            ld = calc_enc_loss(z1_flat, z2_flat, global_step, z3=z3_flat, deltas=d_expanded, target=t_expanded,
                               non_emptys=ne_flat, psize=psize, loss_weights=lw)
        else:
            ld = {'loss': torch.tensor(0.0, device=z1_l.device, dtype=z1_l.dtype)}
        level_losses.append({k: v.detach() if hasattr(v, 'item') else v for k, v in ld.items()})
        for k, v in ld.items(): total[k] = total.get(k, 0) + v

    if not total: return {'loss': torch.tensor(0.0, device=deltas.device)}
    total = {k: v / n_levels for k, v in total.items()}
    total['levels'] = level_losses
    return total


def calc_dec_loss(decoder, enc_out, img_real, pos_weight=1.0, note_weights=None, lambda_mse=0.2):
    "Decoder loss: BCE (class-imbalance-weighted) + a little MSE."
    img_recon = decoder(enc_out)
    pos_weight_t = torch.tensor([pos_weight], device=img_real.device)
    loss_bce = F.binary_cross_entropy_with_logits(img_recon, img_real, pos_weight=pos_weight_t, weight=note_weights)
    img_recon = torch.sigmoid(img_recon)
    loss_mse = F.mse_loss(img_recon, img_real) if lambda_mse > 0 else 0.0
    loss_dec = loss_bce + lambda_mse * loss_mse
    mse_val = loss_mse.item() if hasattr(loss_mse, 'item') else loss_mse
    return {'dec': loss_dec, 'bce': loss_bce.item(), 'recon': img_recon.detach(), 'mse': mse_val}
