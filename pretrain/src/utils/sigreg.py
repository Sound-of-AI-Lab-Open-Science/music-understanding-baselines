"""SIGReg anti-collapse loss for Music-JEPA (LeJEPA, Balestriero & LeCun).

Drop-in replacement for the VICReg variance/covariance anti-collapse terms in
``src/utils/vicreg.py``.  Reference: **LeJEPA: Provable and Scalable
Self-Supervised Learning Without the Heuristics** (Balestriero & LeCun,
arXiv:2511.08544), Algorithm 1 (``SIGReg``) and Algorithm 2 / Eq. (9).

What SIGReg does
----------------
SIGReg (Sketched Isotropic Gaussian Regularization) drives the embedding
distribution toward an isotropic Gaussian by a *sliced* goodness-of-fit test.
It projects the embeddings onto ``num_slices`` random unit directions on the
sphere and, for each 1-D projection, measures how far its distribution is from a
standard normal ``N(0, 1)`` using an **Epps-Pulley** test statistic (an
integrated, weighted squared difference between the empirical characteristic
function and the Gaussian characteristic function ``phi(t) = exp(-t^2/2)``).
A distribution that is collapsed / degenerate / anisotropic yields a large
statistic; a truly isotropic-Gaussian one yields a small statistic.  Minimizing
the mean statistic therefore both prevents collapse (like VICReg's variance
term) and decorrelates dimensions (like VICReg's covariance term), while
additionally matching *higher moments* -- all in a single quantity.

Faithful reproduction of the paper's Algorithm 1
-------------------------------------------------
::

    def SIGReg(x, global_step, num_slices=1024):        # x: [N, K]
        g = torch.Generator(device=x.device); g.manual_seed(global_step)
        A = torch.randn((x.size(1), num_slices), generator=g, device=x.device)
        A = A / A.norm(p=2, dim=0)                       # unit sphere directions
        t = torch.linspace(-5, 5, 17, device=x.device)  # 17 integration nodes
        exp_f = torch.exp(-0.5 * t**2)                   # phi(t) = exp(-t^2/2)
        x_t = (x @ A).unsqueeze(2) * t                   # [N, num_slices, 17]
        ecf = (1j * x_t).exp().mean(0)                   # empirical CF [S, 17]
        # ecf = all_reduce(ecf, "AVG")                   # multi-GPU only (skipped)
        err = (ecf - exp_f).abs().square().mul(exp_f)    # |phi_hat-phi|^2 * w(t)
        N = x.size(0)
        T = torch.trapz(err, t, dim=1) * N               # [num_slices]
        return T                                          # per-slice statistics

    L = (1 - lambda) * L_invariance + lambda * mean(SIGReg)     # Eq. (9), lambda=0.05

Adaptation to the Music-JEPA codebase
-------------------------------------
* We apply SIGReg to the **predictor output** ``yhat`` (shape ``[M, d]``, ``M`` =
  number of flattened valid prediction targets), exactly where the VICReg
  var/cov terms were applied.  In LeJEPA the regularizer is applied to the
  *encoder embeddings across views*; here the anti-collapse surface is ``yhat``,
  consistent with this repo's existing convention.
* The invariance / prediction term stays the **cosine regression** loss
  ``L_cos = mean(1 - cos(yhat_i, y_i))`` (paper Hachana & Rasheed), NOT LeJEPA's
  L2 prediction loss.  We only swap the anti-collapse family.
* **Single-GPU**: the ``all_reduce`` of the empirical CF and the ``world_size``
  factor are no-ops and are omitted (this repo trains ``devices: 1``).

Numerical requirements
----------------------
* The complex ``exp`` and the characteristic-function algebra run in **float32**
  inside an explicit ``torch.autocast(enabled=False)`` region -- training uses
  ``bf16-mixed`` and complex ops + stable statistics need fp32.  This mirrors how
  ``variance_covariance`` forces float32.
* ``global_step`` seeds the random projection directions so the slice set is
  deterministic within a step (and rotates across steps), per Algorithm 1.

Magnitude / normalization
-------------------------
The raw per-slice statistic is multiplied by ``N`` (= ``M``, the number of
target rows), so ``mean(SIGReg)`` grows ~linearly with the batch's target count
(here often hundreds-to-thousands).  That makes the *raw* statistic far larger
than ``L_cos`` (~0.2-1.0).  ``normalize='per_n'`` (the default) divides ``T`` by
``N`` so the loss is O(1) and combines sanely with ``lambda = 0.05``; the raw
form is kept available via ``normalize='raw'`` for faithfulness / diagnostics.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def sigreg_term(
    yhat: torch.Tensor,
    global_step: int = 0,
    num_slices: int = 1024,
    num_points: int = 17,
    domain: float = 5.0,
    normalize: str = "per_n",
) -> torch.Tensor:
    """SIGReg anti-collapse statistic on ``yhat`` (``[N, K]``) -> scalar.

    Faithful implementation of LeJEPA Algorithm 1 (arXiv:2511.08544): project
    ``yhat`` onto ``num_slices`` random unit directions, compute the Epps-Pulley
    per-slice Gaussianity statistic against ``N(0, 1)``, and return the mean over
    slices.

    Parameters
    ----------
    yhat : Tensor ``[N, K]``
        Rows are samples, columns are embedding dimensions.
    global_step : int
        Seeds the random projection generator (deterministic within a step).
    num_slices : int
        Number of random 1-D projections (paper default 1024).
    num_points : int
        Number of integration nodes for the CF integral (paper default 17).
    domain : float
        Integration domain ``[-domain, domain]`` for ``t`` (paper default 5.0).
    normalize : {'per_n', 'raw'}
        ``'per_n'`` (default) divides the per-slice statistic by ``N`` so the
        result is O(1) -- recommended when combining with ``lambda = 0.05``.
        ``'raw'`` keeps the paper's ``* N`` factor verbatim.

    Returns
    -------
    Tensor scalar
        ``mean`` over slices of the per-slice statistic.  Zero (grad-safe) if
        ``N < 2`` (the empirical CF of a single row is meaningless).
    """
    if normalize not in ("per_n", "raw"):
        raise ValueError("normalize must be 'per_n' or 'raw', got {!r}".format(normalize))

    n = yhat.shape[0]
    if n < 2:
        # Undefined for a single row; return a grad-safe zero.
        return yhat.sum() * 0.0

    # Complex CF math + goodness-of-fit stats must run in fp32, never under
    # bf16-mixed autocast (mirrors variance_covariance's float() coercion).
    with torch.autocast(device_type=yhat.device.type, enabled=False):
        x = yhat.float()
        k = x.shape[1]

        # Seeded random unit directions on the sphere: A [K, num_slices].
        g = torch.Generator(device=x.device)
        g.manual_seed(int(global_step))
        A = torch.randn((k, num_slices), generator=g, device=x.device, dtype=torch.float32)
        A = A / A.norm(p=2, dim=0, keepdim=True)

        # Integration nodes and the target Gaussian CF / weight.
        t = torch.linspace(-domain, domain, num_points, device=x.device, dtype=torch.float32)
        exp_f = torch.exp(-0.5 * t.pow(2))                      # phi(t)=exp(-t^2/2) [P]

        # Projections -> per-node phase argument: [N, num_slices, P].
        x_t = (x @ A).unsqueeze(2) * t
        # Empirical characteristic function per slice/node: mean over samples.
        ecf = (1j * x_t).exp().mean(0)                          # [num_slices, P] complex
        # (single-GPU: all_reduce(ecf, "AVG") is a no-op and omitted)

        # Weighted squared CF discrepancy vs. the standard-normal CF.
        err = (ecf - exp_f).abs().square().mul(exp_f)           # [num_slices, P] real
        stat = torch.trapz(err, t, dim=1)                       # [num_slices] Epps-Pulley
        stat = stat * float(n)                                  # paper's * N factor

        if normalize == "per_n":
            stat = stat / float(n)                              # -> O(1), lambda-friendly

        return stat.mean()


def _per_dim_variance_monitor(yhat: torch.Tensor, eps: float = 1e-4) -> Dict[str, torch.Tensor]:
    """Reuse the VICReg collapse monitor (per-dim variance + mean std) on ``yhat``.

    Kept identical to ``variance_covariance`` so ``train/perdim_var`` /
    ``train/mean_std`` stay comparable across the vicreg and sigreg loss paths.
    """
    yhat = yhat.float()
    m = yhat.shape[0]
    if m < 2:
        z = yhat.new_zeros(())
        return {"per_dim_variance": z, "mean_std": z}
    centered = yhat - yhat.mean(dim=0, keepdim=True)
    var = centered.pow(2).sum(dim=0) / (m - 1)                  # unbiased [d]
    std = torch.sqrt(var + eps)
    return {
        "per_dim_variance": var.mean().detach(),
        "mean_std": std.mean().detach(),
    }


def sigreg_jepa_loss(
    yhat: torch.Tensor,
    y: torch.Tensor,
    lambd: float = 0.05,
    global_step: int = 0,
    cos_eps: float = 1e-8,
    num_slices: int = 1024,
    num_points: int = 17,
    domain: float = 5.0,
    normalize: str = "per_n",
    var_eps: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    """LeJEPA-style Music-JEPA objective: cosine invariance + SIGReg anti-collapse.

    ::

        total = (1 - lambd) * L_cos  +  lambd * mean(SIGReg(yhat))

    ``L_cos`` is the (unchanged) cosine regression toward the stop-gradient target
    (paper Hachana & Rasheed); the VICReg var/cov anti-collapse terms are replaced
    by SIGReg (LeJEPA, arXiv:2511.08544, Algorithm 1 / Eq. 9, default ``lambda =
    0.05``).

    Returns a dict sharing the existing ``jepa_loss`` keys where sensible so the
    LightningModule logging stays consistent::

        loss              : (1 - lambd)*L_cos + lambd*SIGReg   (optimized)
        loss_cos          : raw cosine regression term
        jepa_loss         : (1 - lambd) * L_cos  (invariance band, paper Fig. 2)
        sigreg_loss       : lambd * mean(SIGReg) (anti-collapse band)
        sigreg_raw        : mean(SIGReg) before the lambd weight (monitor)
        per_dim_variance  : mean_j Var(yhat_.j)  (collapse monitor, Fig. 4)
        mean_std          : mean_j std_j         (collapse monitor)
    """
    loss_cos = F.cosine_similarity(
        yhat.float(), y.float().detach(), dim=-1, eps=cos_eps
    )
    loss_cos = (1.0 - loss_cos).mean()

    sigreg_raw = sigreg_term(
        yhat, global_step=global_step, num_slices=num_slices,
        num_points=num_points, domain=domain, normalize=normalize,
    )

    invariance = (1.0 - lambd) * loss_cos
    sigreg = lambd * sigreg_raw
    total = invariance + sigreg

    mon = _per_dim_variance_monitor(yhat, eps=var_eps)
    return {
        "loss": total,
        "loss_cos": loss_cos,
        "jepa_loss": invariance.detach(),
        "sigreg_loss": sigreg.detach(),
        "sigreg_raw": sigreg_raw.detach(),
        "per_dim_variance": mon["per_dim_variance"],
        "mean_std": mon["mean_std"],
    }
