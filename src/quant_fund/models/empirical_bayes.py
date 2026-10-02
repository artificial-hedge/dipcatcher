"""Empirical-Bayes canon: Robbins' Poisson
nonparametric EB, Tweedie's f-modeling posterior
mean, and Kiefer-Wolfowitz NPMLE g-modeling via
EM on a fixed grid.

`bench_eb` checks Robbins against the naive
estimator on a mixture of Poisson rates, and that
KW-NPMLE recovers a two-point mixing mass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quant_fund._typing import FloatArray

__all__ = [
    "robbins_poisson",
    "tweedie_mean",
    "kw_npmle",
    "bench_eb",
]


def robbins_poisson(x: FloatArray, smooth: bool = True) -> FloatArray:
    """Robbins (1956) nonparametric EB for Poisson:
    E[λ|x] = (x+1)·N(x+1)/N(x). The raw counts make
    the ratio unstable in sparse tails — Maritz's
    fix smooths N with a short triangular kernel
    before forming the ratio."""
    x = np.asarray(x, dtype=np.int64)
    k_max = int(x.max()) + 3
    counts = np.bincount(x, minlength=k_max).astype(np.float64)
    if smooth:
        kern = np.array([0.25, 0.5, 0.25])
        counts = np.convolve(counts, kern, mode="same")
    est = (x + 1) * counts[x + 1] / np.maximum(counts[x], 1e-9)
    return est


def tweedie_mean(z: FloatArray, sigma: float = 1.0, deg: int = 3) -> dict[str, object]:
    """Tweedie f-modeling (Efron 2011): fit log
    marginal density f(z) by Poisson GLM on binned
    counts, then E[μ|z] = z + σ²·(log f)'(z)."""
    z = np.asarray(z, dtype=np.float64)
    n = z.size
    n_bins = max(20, int(np.sqrt(n)))
    edges = np.linspace(z.min(), z.max(), n_bins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(z, bins=edges)
    width = edges[1] - edges[0]
    # Poisson GLM: counts ~ Poisson(n·width·f(ctr))
    # log-mean poly basis
    deg = min(deg, n_bins - 2)
    basis = np.polynomial.legendre.legvander(2 * (ctr - ctr.mean()) / (ctr.max() - ctr.min()), deg)
    offset = np.log(n * width)
    beta = np.zeros(deg + 1)
    for _ in range(100):
        eta = basis @ beta + offset
        mu_p = np.exp(np.clip(eta, -30, 30))
        w = mu_p
        score = basis.T @ (counts - mu_p)
        info = basis.T @ (basis * w[:, None])
        step = np.linalg.solve(info + 1e-8 * np.eye(deg + 1), score)
        beta += step
        if np.linalg.norm(step) < 1e-8:
            break
    # E[mu|z] = z + sigma^2 * (d/dz) log f(z)
    scale = 2.0 / (ctr.max() - ctr.min())
    dlogf = np.zeros(n)
    for j in range(1, deg + 1):
        leg_p = np.polynomial.legendre.legder(np.eye(deg + 1)[j])
        dp = np.polynomial.legendre.legval(2 * (z - ctr.mean()) / (ctr.max() - ctr.min()), leg_p)
        dlogf += beta[j] * dp * scale
    post = z + sigma**2 * dlogf
    return {"post_mean": post, "beta": beta}


def kw_npmle(
    x: FloatArray,
    grid: FloatArray | None = None,
    sigma: float = 1.0,
    it: int = 500,
    tol: float = 1e-7,
) -> dict[str, object]:
    """Kiefer-Wolfowitz (1956) NPMLE for normal
    means: max-marginal-likelihood mixing measure via
    EM on a fixed grid."""
    x = np.asarray(x, dtype=np.float64)
    if grid is None:
        lo, hi = np.percentile(x, [1, 99])
        span = hi - lo
        grid = np.linspace(lo - 0.1 * span, hi + 0.1 * span, 80)
    grid = np.asarray(grid, dtype=np.float64)
    m = grid.size
    # kernel matrix p_ij = phi((x_i - g_j)/sigma)
    d = (x[:, None] - grid[None, :]) / sigma
    kern = np.exp(-0.5 * d**2) / (sigma * np.sqrt(2 * np.pi))
    w = np.full(m, 1.0 / m)
    ll_prev = -np.inf
    for _ in range(it):
        marg = kern @ w
        post = kern * w[None, :] / np.maximum(marg[:, None], 1e-300)
        w_new = post.mean(axis=0)
        w_new = np.maximum(w_new, 0.0)
        w_new /= w_new.sum()
        ll = float(np.mean(np.log(np.maximum(marg, 1e-300))))
        w = w_new
        if abs(ll - ll_prev) < tol:
            break
        ll_prev = ll
    post_mean = post @ grid
    return {"grid": grid, "weights": w, "post_mean": post_mean}


def bench_eb(seed: int = 538) -> dict[str, float]:
    """SYNTHETIC: Robbins beats naive on two-rate
    Poisson; KW-NPMLE recovers a 2-point normal
    mixture; Tweedie shrinks extremes."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    # Robbins Poisson: half λ=2, half λ=8
    lam = np.repeat([2.0, 8.0], 250)
    x_p = rng.poisson(lam)
    naive_mse = float(np.mean((x_p - lam) ** 2))
    eb = robbins_poisson(x_p)
    eb_mse = float(np.mean((eb - lam) ** 2))
    out["synthetic_robbins_ratio"] = eb_mse / naive_mse
    if eb_mse >= naive_mse:
        raise ValueError(f"robbins off: {eb_mse} vs {naive_mse}")
    # KW NPMLE: normal means from two-point prior
    mus = np.repeat([-2.0, 3.0], 200)
    xn = mus + rng.normal(size=400)
    kw = kw_npmle(xn, sigma=1.0, it=400)
    w = np.asarray(kw["weights"])
    g = np.asarray(kw["grid"])
    top2 = g[np.argsort(w)[-2:]]
    mass = float(np.sort(w)[-2:].sum())
    out["synthetic_kw_top2_mass"] = mass
    out["synthetic_kw_modes_dist"] = float(
        np.min(np.abs(top2[0] - np.array([-2.0, 3.0])))
        + np.min(np.abs(top2[1] - np.array([-2.0, 3.0])))
    )
    if out["synthetic_kw_modes_dist"] > 1.0:
        raise ValueError(f"kw modes off: {out['synthetic_kw_modes_dist']:.2f}")
    # Tweedie shrinkage: extreme z pulled toward 0
    zs = np.concatenate([rng.normal(size=300), np.array([4.0, -4.0])])
    tw = tweedie_mean(zs, sigma=1.0, deg=3)
    pm = np.asarray(tw["post_mean"])
    out["synthetic_tweedie_extreme_shrink"] = float(abs(pm[-2]) + abs(pm[-1]))
    if abs(pm[-2]) >= 4.0 or abs(pm[-1]) >= 4.0:
        raise ValueError("tweedie did not shrink extremes")
    return out
