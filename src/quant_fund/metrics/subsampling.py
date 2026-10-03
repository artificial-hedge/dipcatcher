"""Subsampling inference — Politis-Romano-Wolf confidence sets.

Approximate the sampling distribution of a statistic by
recomputing it on every contiguous block of size b < n. The
empirical quantiles of the recentered subsample statistics give
confidence intervals valid under minimal assumptions — including
heavy tails and dependent data where the iid bootstrap fails.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure interval coverage and width
on generated series — never market evidence.

References:
- Politis, D. N., Romano, J. P. (1994). Large sample confidence
  regions based on subsamples under minimal assumptions.
  *Annals of Statistics* 22, 2031-2050.
- Politis, D. N., Romano, J. P., Wolf, M. (1999).
  *Subsampling*, Springer — block choice and τ_n scaling.
- Politis, D. N., Romano, J. P. (1994). The stationary
  bootstrap. *JASA* 89 — contrast for dependent data.
- Bertail, P. (1997). Second-order properties of an extrapolated
  bootstrap without replacement. *Bernoulli* 3.

Composition: pure numpy — overlapping contiguous blocks, root-τ
recentering, quantile inversion; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def subsampling_ci(
    stat: Callable[[FloatArray], float],
    x: FloatArray,
    level: float = 0.9,
    block_frac: float = 0.18,
    max_blocks: int = 400,
    seed: int = 7,
) -> dict[str, float]:
    """Subsampling CI for a scalar statistic on a (T × d) series.

    ``stat`` maps a (b × d) block to a scalar (e.g. OLS slope,
    mean). Blocks are contiguous (respects dependence); if the
    count exceeds ``max_blocks`` a deterministic subset is drawn.
    Returns the equal-tailed interval and diagnostics."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2:
        raise ValueError("x must be (T × d)")
    t, d = xx.shape
    if t < 80 or d < 1 or d > 8:
        raise ValueError("T>=80, d in 1..8")
    if not (0.6 <= level <= 0.99):
        raise ValueError("level in .6..99")
    if not (0.1 <= block_frac <= 0.7):
        raise ValueError("block_frac in .1..7")
    if not np.all(np.isfinite(xx)):
        raise ValueError("finite inputs required")

    b = max(20, int(t * block_frac))
    if b >= t - 10:
        raise ValueError("block too large")
    starts = np.arange(0, t - b + 1)
    if starts.size > max_blocks:
        rng = np.random.default_rng(seed)
        starts = np.sort(rng.choice(starts, max_blocks, replace=False))
    theta_hat = float(stat(xx))
    if not np.isfinite(theta_hat):
        raise ValueError("statistic returned non-finite value")

    subs = np.array([float(stat(xx[s : s + b])) for s in starts], dtype=np.float64)
    if not np.all(np.isfinite(subs)):
        raise ValueError("statistic failed on a subsample")
    # root-τ recentered distribution: √b (θ_b − θ_n)
    scale = np.sqrt(b / 1.0)
    roots = scale * (subs - theta_hat)
    lo_q = np.quantile(roots, (1 - level) / 2)
    hi_q = np.quantile(roots, 1 - (1 - level) / 2)
    n_scale = np.sqrt(t / 1.0)
    ci_lo = theta_hat - hi_q / n_scale
    ci_hi = theta_hat - lo_q / n_scale

    return {
        "theta": theta_hat,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "ci_width": float(ci_hi - ci_lo),
        "n_blocks": float(starts.size),
        "b": float(b),
        "sub_mean": float(np.mean(subs)),
        "sub_sd": float(np.std(subs)),
    }


def subsampling_coverage(
    stat: Callable[[FloatArray], float],
    sampler: Callable[[np.random.Generator], FloatArray],
    truth: float,
    n_rep: int = 60,
    level: float = 0.9,
    seed: int = 0,
) -> dict[str, float]:
    """Monte-Carlo coverage of the subsampling interval."""
    rng = np.random.default_rng(seed)
    hits = 0
    widths = []
    for r in range(n_rep):
        x = sampler(rng)
        out = subsampling_ci(stat, x, level=level, seed=seed + r)
        lo, hi = float(out["ci_lo"]), float(out["ci_hi"])
        hits += int(lo <= truth <= hi)
        widths.append(hi - lo)
    return {
        "coverage": float(hits / n_rep),
        "mean_width": float(np.mean(widths)),
        "n_rep": float(n_rep),
    }


def _ar1_mean(x: FloatArray) -> float:
    """AR(1) coefficient as the statistic (scalar)."""
    y = x[:, 0]
    y0, y1 = y[:-1], y[1:]
    return float(np.sum(y0 * y1) / max(np.sum(y0 * y0), 1e-12))


def _ar1_sampler(rho: float, t: int) -> Callable[[np.random.Generator], FloatArray]:
    def draw(rng: np.random.Generator) -> FloatArray:
        e = rng.normal(0.0, 1.0, t)
        y = np.zeros(t)
        for i in range(1, t):
            y[i] = rho * y[i - 1] + e[i]
        return y.reshape(-1, 1)

    return draw


def bench_subsampling(seed: int = 20261231 + 232) -> dict[str, float]:
    """Subsampling self-check: CI brackets the AR(1) coefficient
    and Monte-Carlo coverage sits near nominal. All ``synthetic_*``."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, 400)
    y = np.zeros(400)
    for i in range(1, 400):
        y[i] = 0.6 * y[i - 1] + e[i]
    x = y.reshape(-1, 1)
    out = subsampling_ci(_ar1_mean, x, level=0.9, seed=seed)
    cov = subsampling_coverage(_ar1_mean, _ar1_sampler(0.6, 300), 0.6, n_rep=50, seed=seed + 1)
    out_b = subsampling_ci(_ar1_mean, x, level=0.9, seed=seed)
    # asymptotic reference: ρ̂ ± z·√((1-ρ²)/T)
    z = float(stats.norm.ppf(0.95))
    rho = float(out["theta"])
    asym_w = 2 * z * np.sqrt(max(1e-9, 1 - rho * rho) / 400)

    return {
        "synthetic_theta": rho,
        "synthetic_theta_err": float(abs(rho - 0.6)),
        "synthetic_ci_lo": float(out["ci_lo"]),
        "synthetic_ci_hi": float(out["ci_hi"]),
        "synthetic_covers": float(float(out["ci_lo"]) <= 0.6 <= float(out["ci_hi"])),
        "synthetic_coverage_mc": float(cov["coverage"]),
        "synthetic_width_vs_asym": float(out["ci_width"]) / asym_w,
        "synthetic_n_blocks": float(out["n_blocks"]),
        "synthetic_detects": float(
            abs(rho - 0.6) < 0.08
            and float(out["ci_lo"]) <= 0.6 <= float(out["ci_hi"])
            and 0.7 <= float(cov["coverage"]) <= 1.0
        ),
        "synthetic_determinism": float(float(out["ci_lo"]) == float(out_b["ci_lo"])),
    }
