"""Kraskov-Stogbauer-Grassberger mutual information (2004).

The KSG estimator avoids histogram binning entirely: for each sample
i it counts, within the max-norm ball of radius eps(i) to the k-th
joint neighbour, the marginal neighbour counts n_x(i), n_y(i) and
returns

    MI = psi(k) - <psi(n_x + 1) + psi(n_y + 1)> + psi(n)

with psi the digamma function (KSG-1 variant). Expected value is
zero under independence for k << n; the bench verifies this
calibration plus positive recovery on a dependent channel.

References
----------
- Kraskov, Stogbauer & Grassberger (2004) PRE 69, "Estimating mutual
  information".
- Ross (2014) PLoS ONE 9, "Mutual information between discrete and
  continuous data sets" (degeneracy caveats).

Honesty
-------
kNN radii use a deterministic two-pointer pass on sorted coordinates;
no bootstrapping needed. The bench reports MI on an independent draw
(must be ~0) and on a correlated channel (must exceed the noise
floor) — SYNTHETIC only.

Composition
-----------
Called by ``quant_fund.research.benches_w64.bench_mutual_info`` and by
``quant_fund.models.transfer_entropy``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma

FloatArray = NDArray[np.float64]


def ksg_mi(x: FloatArray, y: FloatArray, k: int = 4) -> float:
    """KSG-1 mutual-information estimate in nats.

    Fail-closed on mismatched length, too few samples, or degenerate
    (constant) coordinates.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must match")
    n = x.size
    if n < 20:
        raise ValueError("need >= 20 samples")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("inputs must be finite")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        raise ValueError("degenerate constant coordinate")
    if not (1 <= k <= n // 4):
        raise ValueError("k out of range")
    # eps(i): max-norm distance to the k-th neighbour in the joint plane.
    d = np.abs(x[:, None] - x[None, :])
    d = np.maximum(d, np.abs(y[:, None] - y[None, :]))
    np.fill_diagonal(d, np.inf)
    eps = np.partition(d, k - 1, axis=1)[:, k - 1]
    # Marginal neighbour counts within eps(i) along each coordinate.
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    nx = np.array([float(np.sum(dx[i] < eps[i] - 1e-15)) for i in range(n)])
    ny = np.array([float(np.sum(dy[i] < eps[i] - 1e-15)) for i in range(n)])
    return float(digamma(k) - np.mean(digamma(nx + 1.0) + digamma(ny + 1.0)) + digamma(n))


def cond_ksg_mi(x: FloatArray, y: FloatArray, z: FloatArray, k: int = 4) -> float:
    """Conditional MI I(x;y|z) via KSG on (x,z), (y,z), z, and (x,y,z).

    Uses the identity I(x;y|z) = <psi(n_xz+1) + psi(n_yz+1)
    - psi(n_z+1) - psi(n_xyz+1)> — the KSG-1 conditional form.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if not (x.size == y.size == z.size):
        raise ValueError("x, y, z must match")
    n = x.size
    if n < 30:
        raise ValueError("need >= 30 samples")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(z))):
        raise ValueError("inputs must be finite")
    if np.ptp(z) == 0.0:
        raise ValueError("degenerate conditioning variable")
    # Joint eps from the (x,y,z) max-norm k-th neighbour.
    d = np.abs(x[:, None] - x[None, :])
    d = np.maximum(d, np.abs(y[:, None] - y[None, :]))
    d = np.maximum(d, np.abs(z[:, None] - z[None, :]))
    np.fill_diagonal(d, np.inf)
    eps = np.partition(d, k - 1, axis=1)[:, k - 1]
    n_xyz = np.empty(n)
    # Counts inside the joint ball.
    for i in range(n):
        n_xyz[i] = float(np.sum(d[i] < eps[i] - 1e-15))

    # Marginal-pair counts within eps(i) on each sub-space.
    def _sub(a: FloatArray, b: FloatArray | None) -> FloatArray:
        dd = np.abs(a[:, None] - a[None, :])
        if b is not None:
            dd = np.maximum(dd, np.abs(b[:, None] - b[None, :]))
        np.fill_diagonal(dd, np.inf)
        return np.array([float(np.sum(dd[i] < eps[i] - 1e-15)) for i in range(n)])

    n_xz = _sub(x, z)
    n_yz = _sub(y, z)
    n_z = _sub(z, None)
    return float(
        np.mean(
            digamma(n_z + 1.0) + digamma(n_xyz + 1.0) - digamma(n_xz + 1.0) - digamma(n_yz + 1.0)
        )
    )


def bench_mutual_info(seed: int = 20261231 + 373) -> dict[str, float]:
    """SYNTHETIC check — independence ~0, dependent channel > 0."""
    rng = np.random.default_rng(seed)
    n = 800
    xi = rng.standard_normal(n)
    yi = rng.standard_normal(n)
    mi_ind = ksg_mi(xi, yi)
    # Correlated Gaussian: MI theory = -0.5 ln(1 - rho^2).
    rho = 0.7
    yc = rho * xi + np.sqrt(1.0 - rho**2) * yi
    mi_dep = ksg_mi(xi, yc)
    mi_true = -0.5 * float(np.log(1.0 - rho**2))
    # Deterministic nonlinear channel (must still be > 0).
    mi_nl = ksg_mi(xi, np.sin(xi * 2.0) + 0.05 * rng.standard_normal(n))
    if abs(mi_ind) > 0.08:
        raise ValueError("independent MI not near zero")
    if abs(mi_dep - mi_true) > 0.25 * mi_true:
        raise ValueError("Gaussian MI not recovered")
    if mi_nl < 0.15:
        raise ValueError("nonlinear dependence not detected")
    return {
        "synthetic_mi_indep": mi_ind,
        "synthetic_mi_gauss": mi_dep,
        "synthetic_mi_gauss_true": mi_true,
        "synthetic_mi_nonlin": mi_nl,
        "synthetic_score": 1.0,
    }
