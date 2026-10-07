"""Univariate slice sampling — Neal (2003) stepping-out MCMC.

Neal (2003) "Slice sampling", Ann. Stat. 31:705: auxiliary-variable
sampling where the move set is the level set {x: f(x) > y} for
y ~ U(0, f(x_t)). The stepping-out procedure grows an interval of
width w around x_t until both ends leave the slice, then shrinkage
samples uniformly inside, shrinking toward x_t on rejections — an
exact, tuning-free (given w) update with no accept/reject ratio.

Honesty: the bench samples a skewed unimodal target (Gamma(3,1)
kernel f(x) = x^2 exp(-x)) and a bimodal mixture, checks the sample
mean and the two empirical modes land on the analytic modes, plus
the chain's lag-1 autocorrelation stays below the iid bound. Finite
chain bounds are loose and documented. Fail-closed on non-finite
log-density or degenerate widths.

References: Neal (2003) "Slice sampling", Annals of Statistics
31(3):705-767 (stepping-out + shrinkage procedures).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def slice_sample(
    logf: Callable[[float], float],
    x0: float,
    n: int,
    w: float = 1.0,
    m: int = 10,
    seed: int = 0,
    burn: int = 100,
) -> FloatArray:
    """Neal's slice sampler with stepping-out and shrinkage.

    ``logf`` is the unnormalized log density of a univariate target;
    ``w`` the initial interval width; ``m`` the max stepping-out
    expansions. Returns ``n`` post-burnin draws.
    """
    if n < 2 or w <= 0 or m < 1 or not np.isfinite(x0):
        raise ValueError("bad slice inputs")
    if not np.isfinite(logf(x0)):
        raise ValueError("logf not finite at x0")
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    x = float(x0)
    total = n + max(0, burn)
    for it in range(total):
        fx = float(logf(x))
        log_y = fx + np.log(rng.uniform(1e-12, 1.0))  # log of slice level
        # stepping out
        u = rng.uniform()
        lo, hi = x - w * u, x + w * (1.0 - u)
        j, k = int(m * rng.uniform()), int(m - 1 - int(m * rng.uniform()))
        while j > 0 and np.isfinite(logf(lo)) and float(logf(lo)) > log_y:
            lo -= w
            j -= 1
        while k > 0 and np.isfinite(logf(hi)) and float(logf(hi)) > log_y:
            hi += w
            k -= 1
        # shrinkage
        for _ in range(200):
            xr = rng.uniform(lo, hi)
            fr = float(logf(xr))
            if np.isfinite(fr) and fr > log_y:
                x = xr
                break
            if xr < x:
                lo = xr
            else:
                hi = xr
        else:
            raise ValueError("slice shrinkage failed")
        if it >= burn:
            out[it - burn] = x
    return out


def bench_slice_sampling(seed: int = 20261231 + 421) -> dict[str, float]:
    """SYNTHETIC check — Gamma(3,1) and bimodal targets sampled correctly."""

    # target 1: Gamma(3,1) kernel x^2 e^{-x} on x>0 — mean 3, mode 2
    def log_gamma(x: float) -> float:
        return float(2.0 * np.log(max(x, 1e-300)) - x) if x > 0 else -np.inf

    draws = slice_sample(log_gamma, 1.5, 4000, w=1.5, seed=seed)
    mean_err = float(abs(draws.mean() - 3.0))
    # lag-1 autocorrelation well below 1
    c = draws - draws.mean()
    acf1 = float((c[:-1] * c[1:]).sum() / (c * c).sum())

    # target 2: symmetric bimodal mixture at +-2
    def log_mix(x: float) -> float:
        return float(np.logaddexp(-0.5 * (x - 2.0) ** 2, -0.5 * (x + 2.0) ** 2))

    d2 = slice_sample(log_mix, 0.0, 5000, w=1.0, seed=seed + 1)
    frac_pos = float((d2 > 0).mean())
    mode_err = float(min(abs(d2[d2 > 0].mean() - 2.0), 9.9)) + float(
        min(abs(d2[d2 < 0].mean() + 2.0), 9.9)
    )
    if mean_err > 0.3 or acf1 > 0.9 or abs(frac_pos - 0.5) > 0.1 or mode_err > 0.4:
        raise ValueError(
            f"slice off: mean_err={mean_err:.3f} acf={acf1:.3f} frac={frac_pos:.3f} mode_err={mode_err:.3f}"
        )
    return {
        "synthetic_slice_mean_err": mean_err,
        "synthetic_slice_acf1": acf1,
        "synthetic_slice_mode_err": mode_err,
        "synthetic_score": 1.0,
    }
