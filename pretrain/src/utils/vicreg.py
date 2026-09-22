"""VICReg-style regularization + cosine regression loss for Music-JEPA.

Implements the three loss terms of the objective in Hachana & Rasheed (§3.3),
each returned *separately* so the training loop can weight and log them
independently::

    L = alpha * L_cos + beta * L_var + sigma * L_cov

    L_cos = (1/N) * sum_i ( 1 - <yhat_i, y_i> / (||yhat_i|| ||y_i||) )   (cosine regression)
    L_var = (1/d) * sum_j max(0, gamma - sqrt(Var(yhat_.j) + eps))       (variance hinge)
    L_cov = (1/d) * sum_{i!=j} Cov(yhat_.i, yhat_.j)^2                    (off-diagonal covariance)

Design / numerical notes
------------------------
* **The paper applies the variance and covariance regularizers to the predictor
  output ``yhat`` only** (not to the target branch ``y``), unlike vanilla VICReg
  which regularizes both branches.  We follow the paper: ``variance_covariance``
  takes a single tensor (``yhat``).
* ``L_cos`` is a *regression* toward the (stop-gradient) target representation,
  so we detach ``y`` inside the loss as a defensive guard even though the target
  encoder already runs under ``no_grad``.
* ``Var`` uses ``sqrt(var + eps)`` (eps default 1e-4) to keep the gradient finite
  as the variance approaches zero -- a bare ``sqrt`` has an infinite derivative
  at 0 and produces NaNs once a dimension collapses, exactly the regime this term
  is meant to escape.
* Covariance uses the unbiased estimator (``/ (M - 1)``) and sums the *squared
  off-diagonal* entries, matching the paper's ``i != j`` sum, then divides by
  ``d`` (the VICReg convention, which the paper reproduces).
* Everything is computed in float32 regardless of the incoming dtype so the
  statistics are stable under ``bf16-mixed`` autocast.
* ``per_dim_variance`` (mean over dims of ``Var(yhat_.j)``) is returned as a
  *monitoring* scalar -- this is the quantity the paper ablation plots
  (Fig. 4, magnitude ~0..12) and the signal that the representation has not
  collapsed.

All functions accept ``yhat`` / ``y`` already flattened to ``[M, d]`` (M = total
number of valid prediction targets in the batch).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return a flat view of the off-diagonal entries of a square matrix ``x``."""
    n, m = x.shape
    assert n == m, "off_diagonal expects a square matrix, got {}".format(tuple(x.shape))
    # standard VICReg trick: drop the last element, reshape to (n-1, n+1), drop
    # the first column -> exactly the off-diagonal entries.
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cosine_regression_loss(
    yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """``L_cos`` = mean over rows of ``1 - cos(yhat_i, y_i)``.

    ``yhat``/``y``: ``[M, d]``.  ``y`` is detached (stop-gradient target).
    Returns a scalar in ``[0, 2]`` (0 = perfectly aligned).
    """
    yhat = yhat.float()
    y = y.float().detach()
    cos = F.cosine_similarity(yhat, y, dim=-1, eps=eps)
    return (1.0 - cos).mean()


def variance_covariance(
    yhat: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4
) -> Dict[str, torch.Tensor]:
    """Compute the VICReg variance + covariance terms on ``yhat`` (``[M, d]``).

    Returns a dict with:
        ``loss_var``        : ``(1/d) sum_j max(0, gamma - std_j)``    (scalar)
        ``loss_cov``        : ``(1/d) sum_{i!=j} Cov_ij^2``           (scalar)
        ``per_dim_variance``: ``mean_j Var(yhat_.j)``                 (monitor)
        ``mean_std``        : ``mean_j std_j``                        (monitor)

    ``M >= 2`` is required for a defined (unbiased) covariance; for ``M < 2`` the
    regularizers are returned as zero (they are undefined / meaningless on a
    single sample) so a degenerate micro-batch cannot crash training.
    """
    yhat = yhat.float()
    m, d = yhat.shape
    if m < 2:
        z = yhat.new_zeros(())
        return {
            "loss_var": z,
            "loss_cov": z,
            "per_dim_variance": z,
            "mean_std": z,
        }

    centered = yhat - yhat.mean(dim=0, keepdim=True)
    # unbiased per-dimension variance
    var = centered.pow(2).sum(dim=0) / (m - 1)          # [d]
    std = torch.sqrt(var + eps)                          # eps-guarded sqrt
    loss_var = torch.clamp(gamma - std, min=0.0).mean()  # (1/d) sum_j hinge

    cov = (centered.T @ centered) / (m - 1)              # [d, d]
    loss_cov = off_diagonal(cov).pow(2).sum() / d        # (1/d) sum_{i!=j} cov^2

    return {
        "loss_var": loss_var,
        "loss_cov": loss_cov,
        "per_dim_variance": var.mean().detach(),
        "mean_std": std.mean().detach(),
    }


def soft_effective_rank(
    features: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Differentiable soft effective rank of a feature matrix ``features`` ``[N, d]``.

    Effective rank = ``exp(H)`` where ``H`` is the Shannon entropy of the
    normalized eigenvalue spectrum of the (unbiased) feature covariance
    ``C = P_c^T P_c / (N - 1)`` -- a smooth, differentiable proxy for the number of
    active directions in the representation (IDRR / Matrix-SSL style).  This is the
    quantity that collapses on the pooled context features (~1.4 collapsed vs ~3.3
    healthy), so penalizing it toward a target directly fights that collapse.

    Gram trick: when ``N <= d`` the nonzero eigenvalues of the ``[d, d]`` covariance
    equal those of the ``[N, N]`` Gram matrix ``P_c P_c^T / (N - 1)``, so we
    eigendecompose the smaller matrix for efficiency.

    Numerics: computed in float32 (stable under ``bf16-mixed``); eigenvalues are
    clamped to ``>= 0`` and floored by ``eps`` before normalization so
    ``log`` never sees a non-positive input.  Gradients flow through ``features``.
    Requires ``N >= 2`` (centering removes one rank); the caller should guard tiny
    batches.  Returns a float32 scalar tensor in ``[1, min(N, d)]``.
    """
    # autocast must be DISABLED here: under bf16-mixed the matmul below would be
    # re-cast to bfloat16 even though the inputs are float32, and
    # linalg_eigvalsh has no bfloat16 CUDA kernel.
    with torch.autocast(device_type=features.device.type, enabled=False):
        P = features.float()
        n, d = P.shape
        Pc = P - P.mean(dim=0, keepdim=True)
        if n <= d:
            gram = (Pc @ Pc.T) / (n - 1)                # [N, N] Gram trick
        else:
            gram = (Pc.T @ Pc) / (n - 1)                # [d, d] covariance
        gram = 0.5 * (gram + gram.T)                     # symmetrize for eigvalsh
        lam = torch.linalg.eigvalsh(gram.float())        # ascending real eigenvalues
    lam = torch.clamp(lam, min=0.0) + eps
    p = lam / lam.sum()
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy)


def jepa_loss(
    yhat: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 25.0,
    sigma: float = 1.0,
    gamma: float = 1.0,
    var_eps: float = 1e-4,
    cos_eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Full Music-JEPA objective over flattened targets ``yhat``/``y`` (``[M, d]``).

    Returns every component plus the weighted totals::

        loss              : alpha*L_cos + beta*L_var + sigma*L_cov  (optimized)
        loss_cos/var/cov  : the three *raw* (unweighted) terms
        jepa_loss         : alpha * L_cos          (paper Fig. 2 "JEPA loss")
        vicreg_loss       : beta*L_var + sigma*L_cov (paper Fig. 2 "VICReg loss")
        per_dim_variance  : monitor (paper Fig. 4)
        mean_std          : monitor
    """
    loss_cos = cosine_regression_loss(yhat, y, eps=cos_eps)
    vc = variance_covariance(yhat, gamma=gamma, eps=var_eps)
    loss_var, loss_cov = vc["loss_var"], vc["loss_cov"]

    jepa = alpha * loss_cos
    vicreg = beta * loss_var + sigma * loss_cov
    total = jepa + vicreg
    return {
        "loss": total,
        "loss_cos": loss_cos,
        "loss_var": loss_var,
        "loss_cov": loss_cov,
        "jepa_loss": jepa.detach(),
        "vicreg_loss": vicreg.detach(),
        "per_dim_variance": vc["per_dim_variance"],
        "mean_std": vc["mean_std"],
    }
