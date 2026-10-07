"""Hasbrouck (1995) information share for price discovery.

References
----------
- Hasbrouck, J. (1995). "One Security, Many Markets:
  Determining the Contributions to Price Discovery."
  *Journal of Finance* 50(4), 1175-1199.
- Gonzalo, J. & Granger, C. (1995). "Estimation of Common
  Long-Memory Components in Cointegrated Systems."
  *Journal of Business & Economic Statistics* 13(1), 27-35.
- Baillie, R.T., Booth, G.G., Tse, Y. & Zabotina, T. (2002).
  "Price Discovery and Common Factor Models." *Journal of
  Financial Markets* 5(3), 309-321.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Two cointegrated price quotes ``p = (p1, p2)`` share one
common efficient-price random walk. Estimate the VEC(0)
error-correction form ``dp = alpha * ect_{t-1} + eps`` with
``ect = p1 - beta * p2``; the Gonzalo-Granger permanent
component loads on ``alpha_orth`` (the vector orthogonal to
the adjustment speeds) — market j's permanent weight is
``w_j = alpha_orth_j / sum(alpha_orth)``. Hasbrouck's
information share attributes the common-walk *variance*:
with long-run impact vector ``psi`` (row of the Beveridge
permanent-impact matrix) and innovation covariance Omega,

    IS_j = psi_j^2 * Omega_jj / (psi' Omega psi)

when innovations are uncorrelated; under correlation the
share is order-dependent, so we report the standard upper/
lower bounds from both Cholesky orderings and their midpoint.
The synth: two quotes tracking one planted random walk where
market 1's transitory noise is small (informed venue) and
market 2's is large — IS_1 must dominate; a symmetric-noise
null splits IS near 0.5 each.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def information_share(
    p: FloatArray,
) -> dict[str, float]:
    """Hasbrouck IS bounds + Gonzalo-Granger permanent weights."""
    pp = np.asarray(p, dtype=np.float64)
    if pp.ndim != 2 or pp.shape[1] != 2 or pp.shape[0] < 100:
        raise ValueError("bad price panel")
    if not np.all(np.isfinite(pp)):
        raise ValueError("non-finite")
    t = pp.shape[0]
    # step 1: cointegrating regression p1 = c + beta p2 + ect
    x = np.column_stack([np.ones(t), pp[:, 1]])
    coef, *_ = np.linalg.lstsq(x, pp[:, 0], rcond=None)
    beta = float(coef[1])
    ect = pp[:, 0] - beta * pp[:, 1] - float(coef[0])
    # step 2: VEC dp = alpha * ect_{t-1} + eps
    dp = np.diff(pp, axis=0)
    z = ect[:-1][:, None]
    alpha = np.empty(2)
    eps = np.empty_like(dp)
    for j in range(2):
        a, *_ = np.linalg.lstsq(z, dp[:, j], rcond=None)
        alpha[j] = float(a[0])
        eps[:, j] = dp[:, j] - z[:, 0] * alpha[j]
    omega = np.cov(eps.T)
    # permanent impact row: psi ∝ alpha_orth (Gonzalo-Granger)
    a_orth = np.array([-alpha[1], alpha[0]])
    if np.abs(a_orth).sum() < 1e-12:
        raise ValueError("no error correction")
    psi = a_orth / a_orth.sum()
    gg_w = psi.copy()

    # Cholesky orderings -> IS bounds
    def _share(om: FloatArray) -> float:
        L = np.linalg.cholesky(om)
        # innovations std ordered: market1 first
        num = float((psi[0] * L[0, 0]) ** 2)
        den = float(psi @ om @ psi)
        return num / den

    is1_lo = float(_share(omega))  # ordering m1 first => m1 lower-ish
    # swap ordering: reverse innovation order then un-reverse
    om_sw = omega[::-1, ::-1]
    L2 = np.linalg.cholesky(om_sw)
    num2 = float((psi[1] * L2[0, 0]) ** 2)
    den = float(psi @ omega @ psi)
    is2_lo = num2 / den
    is1_hi = 1.0 - is2_lo
    return {
        "is1_lo": float(is1_lo),
        "is1_hi": float(is1_hi),
        "is1_mid": float(0.5 * (is1_lo + is1_hi)),
        "gg_w1": float(gg_w[0]),
        "beta": beta,
        "alpha1": float(alpha[0]),
        "alpha2": float(alpha[1]),
    }


def synth_hasbrouck(
    seed: int = 20261231 + 320,
    n: int = 2000,
    s1: float = 0.05,
    s2: float = 0.5,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC: two quotes on one planted random walk."""
    rng = np.random.default_rng(seed)
    w = np.cumsum(rng.normal(0.0, 1.0, n))
    p1 = w + rng.normal(0.0, s1, n)
    p2 = w + rng.normal(0.0, s2, n)
    c = np.column_stack([p1, p2])
    # symmetric null: equal noise
    q1 = w + rng.normal(0.0, 0.3, n)
    q2 = w + rng.normal(0.0, 0.3, n)
    return c, np.column_stack([q1, q2])


def bench_hasbrouck(
    seed: int = 20261231 + 320,
) -> dict[str, float]:
    """Wave-55 self-check: informed venue dominates IS."""
    ci, null = synth_hasbrouck(seed=seed)
    r = information_share(ci)
    rn = information_share(null)
    ok = r["is1_lo"] > 0.6 and rn["is1_mid"] > 0.3 and rn["is1_mid"] < 0.7 and r["gg_w1"] > 0.7
    return {
        "synthetic_is1_mid": r["is1_mid"],
        "synthetic_is1_lo": r["is1_lo"],
        "synthetic_is1_null_mid": rn["is1_mid"],
        "synthetic_gg_w1": r["gg_w1"],
        "synthetic_beta": r["beta"],
        "synthetic_score": float(ok),
    }
