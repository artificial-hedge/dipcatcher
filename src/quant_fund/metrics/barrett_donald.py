"""Barrett-Donald (2003) KS tests for stochastic dominance.

First-order dominance of F over G requires F(z) ≤ G(z)
everywhere; second-order dominance compares the integrated
CDFs. The KS-type statistic sup_z (F̂−Ĝ) (or of the
integrated difference) is bootstrapped under the pooled
sample to get p-values.

Honesty: synthetic benches compare a shifted distribution
against its unshifted base and a crossing alternative —
proper diagnostics, never market evidence.

References:
- Barrett, G. F., Donald, S. G. (2003). Consistent tests for
  stochastic dominance. *Econometrica* 71 — KS statistics
  and bootstrap implementation.
- Davidson, R., Duclos, J.-Y. (2000). Statistical inference
  for stochastic dominance and for the measurement of
  poverty and inequality. *Econometrica* 68 — the grid
  approach.
- Linton, O., Maasoumi, E., Whang, Y.-J. (2005). Consistent
  testing for stochastic dominance under general sampling
  schemes. *Review of Economic Studies* 72 — subsampling.
- McFadden, D. (1989). Testing for stochastic dominance. In
  Fomby, T. B., Seo, T. K. (eds.), *Studies in the
  Economics of Uncertainty* — early formulation.

Composition: pure numpy — ecdf comparisons + bootstrap;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ecdf_diff(x: FloatArray, y: FloatArray, grid: FloatArray) -> FloatArray:
    fx = np.searchsorted(np.sort(x), grid, side="right") / x.size
    fy = np.searchsorted(np.sort(y), grid, side="right") / y.size
    return np.asarray(fx - fy, dtype=np.float64)


def sd_statistic(
    x: FloatArray,
    y: FloatArray,
    order: int = 1,
    n_grid: int = 200,
) -> float:
    """KS statistic for H0: x's distribution dominates y's
    at the given order (1: CDFs, 2: integrated CDFs)."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if order not in (1, 2):
        raise ValueError("order 1 or 2 required")
    if xx.ndim != 1 or yy.ndim != 1 or xx.size < 20 or yy.size < 20:
        raise ValueError("1-D arrays with >=20 obs required")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
        raise ValueError("finite inputs required")
    lo = float(min(xx.min(), yy.min()))
    hi = float(max(xx.max(), yy.max()))
    grid = np.linspace(lo, hi, n_grid)
    d = _ecdf_diff(xx, yy, grid)
    if order == 1:
        stat = float(np.max(d))
    else:
        # integrated CDF diff
        integ = np.cumsum(d) * (grid[1] - grid[0])
        stat = float(np.max(integ))
    scale = float(np.sqrt(xx.size * yy.size / (xx.size + yy.size)))
    return float(stat * scale)


def sd_pvalue(
    x: FloatArray,
    y: FloatArray,
    order: int = 1,
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap p-value: resample x and y from the pooled
    sample (least-favorable point of H0 is F=G)."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.ndim != 1 or yy.ndim != 1 or xx.size < 20 or yy.size < 20:
        raise ValueError("1-D arrays with >=20 obs required")
    if order not in (1, 2):
        raise ValueError("order 1 or 2 required")
    if n_boot < 50:
        raise ValueError("n_boot >= 50 required")
    stat = sd_statistic(xx, yy, order=order)
    pooled = np.concatenate([xx, yy])
    rng = np.random.default_rng(seed)
    reps = np.zeros(n_boot)
    for b in range(n_boot):
        bx = rng.choice(pooled, size=xx.size, replace=True)
        by = rng.choice(pooled, size=yy.size, replace=True)
        reps[b] = sd_statistic(bx, by, order=order)
    p = float(np.mean(reps >= stat))
    return {"stat": stat, "p": p}


def synth_dominance(
    n: int = 400,
    shift: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """x = base + shift stochastically dominates y = base."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n) + shift
    y = rng.normal(0, 1, n)
    return {"x": x, "y": y}


def synth_crossing(
    n: int = 400,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """x has higher mean but fatter tails → CDFs cross near
    the left tail → no first-order dominance either way."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.5, 1.8, n)
    y = rng.normal(0, 1, n)
    return {"x": x, "y": y}


def bench_barrett_donald(seed: int = 20261231 + 262) -> dict[str, float]:
    """BD self-check: clear dominance rejects non-dominance
    (small p for the wrong direction, large for the right);
    crossing CDFs reject both directions' weak claim.
    All ``synthetic_*``."""
    d = synth_dominance(shift=0.6, seed=seed)
    x = np.asarray(d["x"])
    y = np.asarray(d["y"])
    # x dominates y: claim "x dom y" holds (large p);
    # claim "y dom x" is violated (small p)
    p_right = sd_pvalue(x, y, order=1, n_boot=300, seed=seed)
    p_wrong = sd_pvalue(y, x, order=1, n_boot=300, seed=seed + 1)
    dc = synth_crossing(seed=seed + 2)
    p_cross = sd_pvalue(
        np.asarray(dc["x"]),
        np.asarray(dc["y"]),
        order=1,
        n_boot=300,
        seed=seed + 3,
    )
    p2 = sd_pvalue(x, y, order=1, n_boot=300, seed=seed)
    return {
        "synthetic_p_dominant": p_right["p"],
        "synthetic_p_reverse": p_wrong["p"],
        "synthetic_stat_dominant": p_right["stat"],
        "synthetic_p_crossing": p_cross["p"],
        "synthetic_detects": float(
            p_right["p"] > 0.3 and p_wrong["p"] < 0.05 and p_cross["p"] < 0.10
        ),
        "synthetic_determinism": float(p2["p"] == p_right["p"]),
    }
