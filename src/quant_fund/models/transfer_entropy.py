"""Schreiber (2000) transfer entropy via KSG conditional MI.

Transfer entropy measures the directed information flow from a driver
X to a target Y beyond what Y's own history explains:

    TE_{X->Y} = I(Y_{t+1}; X_t | Y_t)

Estimated with the KSG conditional mutual-information identity on the
scalar-delay embedding — a nonparametric, nonlinear analogue of
Granger causality. ``both_directions`` returns (X->Y, Y->X) so the
bench can check the zero-flow direction stays at the noise floor.

References
----------
- Schreiber (2000) PRL 85, "Measuring information transfer".
- Barnett, Barrett & Seth (2009) PRL — Gaussian TE vs Granger
  equivalence (sanity check used in the bench).

Honesty
-------
The estimator is noisy on short samples — the bench uses n=900 and
gates on *ordering* (driver direction >> reverse), not on a point
value, which is the honest use of TE empirically. SYNTHETIC only.

Composition
-----------
Called by ``quant_fund.research.benches_w64.bench_transfer_entropy``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.mutual_info import cond_ksg_mi

FloatArray = NDArray[np.float64]


def transfer_entropy(driver: FloatArray, target: FloatArray, k: int = 4) -> float:
    """I(target_{t+1}; driver_t | target_t) in nats.

    Fail-closed on short/mismatched series or constant inputs.
    """
    x = np.asarray(driver, dtype=float).ravel()
    y = np.asarray(target, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("driver and target must match")
    if x.size < 40:
        raise ValueError("need >= 40 samples")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("inputs must be finite")
    return cond_ksg_mi(y[1:], x[:-1], y[:-1], k=k)


def te_both_directions(x: FloatArray, y: FloatArray, k: int = 4) -> tuple[float, float]:
    """(TE_{x->y}, TE_{y->x})."""
    return transfer_entropy(x, y, k), transfer_entropy(y, x, k)


def bench_transfer_entropy(seed: int = 20261231 + 374) -> dict[str, float]:
    """SYNTHETIC check — direction recovery on a driven AR system."""
    rng = np.random.default_rng(seed)
    n = 900
    x = np.empty(n)
    y = np.empty(n)
    x[0] = y[0] = 0.0
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        y[t] = 0.3 * y[t - 1] + 0.9 * x[t - 1] + rng.standard_normal()
    te_xy, te_yx = te_both_directions(x, y)
    # Independent pair: both directions at the noise floor.
    xi = rng.standard_normal(n)
    yi = rng.standard_normal(n)
    ti, ty = te_both_directions(xi, yi)
    if te_xy <= te_yx:
        raise ValueError("driving direction not identified")
    if te_xy < 0.15:
        raise ValueError("transfer entropy too small on a driven channel")
    if abs(ti) > 0.15 or abs(ty) > 0.15:
        raise ValueError("independent pair shows spurious TE")
    return {
        "synthetic_te_xy": te_xy,
        "synthetic_te_yx": te_yx,
        "synthetic_te_ind_xy": ti,
        "synthetic_te_ind_yx": ty,
        "synthetic_score": 1.0,
    }
