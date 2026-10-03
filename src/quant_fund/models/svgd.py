"""Stein variational gradient descent — Liu & Wang (2016).

SVGD transports a set of particles {x_i} toward a target density p
by iterative functional gradient descent in the RKHS unit ball:

    phi*(x) = (1/n) sum_j [ k(x_j, x) grad log p(x_j)
                           + grad_{x_j} k(x_j, x) ]

    x_i <- x_i + eta * phi*(x_i)

The first term attracts particles along the score; the kernel
repulsion term pushes them apart, so the ensemble approximates p —
deterministic, parallel-friendly posterior sampling without MCMC
chains. With the RBF kernel the bandwidth is the median heuristic.

References
----------
- Liu, Q., Wang, D. (2016). "Stein variational gradient descent: a
  general purpose Bayesian inference algorithm." *NeurIPS* 29.
- Liu, Q. (2017). "Stein variational gradient descent as gradient
  flow." *NeurIPS* 30 — the gradient-flow perspective.
- Gorham, J., Mackey, L. (2017). "Measuring sample quality with
  kernels." *ICML* — the KSD diagnostic this transports for.

Honesty
-------
SYNTHETIC targets only; bench verifies particles recover a known
posterior mean/variance and both modes of a bimodal target — not a
live-posterior claim.

Composition
-----------
Called by ``quant_fund.research.benches_w66.bench_svgd``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ScoreFn = Callable[[FloatArray], FloatArray]  # (n,d) -> (n,d) grad log p


def _median_bandwidth(x: FloatArray) -> float:
    """Median heuristic h = med_d2 / log(n+1) with kernel
    exp(-d^2/h) — the Liu & Wang (2016) convention."""
    d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    return float(med / np.log(x.shape[0] + 1.0))


def svgd(
    x0: FloatArray,
    grad_logp: ScoreFn,
    n_iter: int = 500,
    eta: float = 0.1,
    bandwidth: float | None = None,
) -> FloatArray:
    """Run SVGD on particles ``x0`` (n, d); returns final particles."""
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("need (n>=2, d) particles")
    if not (n_iter >= 1 and eta > 0):
        raise ValueError("bad iteration spec")
    n = x.shape[0]
    for _ in range(n_iter):
        h2 = bandwidth if bandwidth is not None else _median_bandwidth(x)
        if h2 <= 0:
            raise ValueError("degenerate bandwidth")
        diff = x[:, None, :] - x[None, :, :]  # (n,n,d)
        k = np.exp(-np.sum(diff * diff, axis=-1) / h2)  # (n,n)
        g = np.asarray(grad_logp(x), dtype=float)  # (n,d)
        attract = (k @ g) / n
        # grad_{x_j} k(x_j,x) = -2 k (x_j - x)/h summed over j:
        repel = -(k[:, :, None] * diff).sum(axis=0) * 2.0 / (n * h2)
        x = x + eta * (attract + repel)
    return x


def bench_svgd(seed: int = 20261231 + 389) -> dict[str, float]:
    """SYNTHETIC check — particles recover Gaussian and bimodal p."""
    rng = np.random.default_rng(seed)
    n, d = 300, 2
    # Target 1: N(mu, Sigma) diagonal.
    mu = np.array([1.5, -0.7])
    sd = np.array([0.8, 0.5])
    prec = 1.0 / (sd * sd)

    def score_gauss(x: FloatArray) -> FloatArray:
        return np.asarray(-(x - mu) * prec, dtype=np.float64)

    x0 = rng.standard_normal((n, d)) * 3.0
    xf = svgd(x0, score_gauss, n_iter=500, eta=1.0)
    mean_err = float(np.linalg.norm(xf.mean(axis=0) - mu))
    var_err = float(np.linalg.norm(xf.var(axis=0) - sd * sd) / np.linalg.norm(sd * sd))
    if mean_err > 0.1 or var_err > 0.3:
        raise ValueError("Gaussian posterior recovery failed")
    # Target 2: symmetric bimodal mixture N(+-2, 0.4^2) on the x-axis.
    mix_sd = 0.4
    mode = 2.0

    def score_mix(x: FloatArray) -> FloatArray:
        # grad log of 0.5 N(m,I s2) + 0.5 N(-m,I s2) on first coord
        z1 = np.exp(-0.5 * ((x[:, 0] - mode) / mix_sd) ** 2)
        z2 = np.exp(-0.5 * ((x[:, 0] + mode) / mix_sd) ** 2)
        w1 = z1 / np.maximum(z1 + z2, 1e-300)
        g0 = w1 * (-(x[:, 0] - mode) / mix_sd**2) + (1 - w1) * (-(x[:, 0] + mode) / mix_sd**2)
        g1 = -x[:, 1] / mix_sd**2
        return np.column_stack([g0, g1])

    y0 = rng.standard_normal((n, d)) * 3.0
    yf = svgd(y0, score_mix, n_iter=800, eta=0.5)
    frac_right = float(np.mean(yf[:, 0] > 0.0))
    if not (0.35 < frac_right < 0.65):
        raise ValueError("bimodal split failed")
    spread = float(np.std(yf[:, 1]))
    if spread < 0.2 or spread > 0.8:
        raise ValueError("component spread off target")
    return {
        "synthetic_svgd_mean_err": mean_err,
        "synthetic_svgd_var_err": var_err,
        "synthetic_svgd_right_frac": frac_right,
        "synthetic_svgd_comp_sd": spread,
        "score": 1.0,
    }
