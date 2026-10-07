"""BDS (Brock-Dechert-Scheinkman-LeBaron 1996) independence test.

References
----------
- Brock, W.A., Dechert, W.D., Scheinkman, J.A. & LeBaron, B.
  (1996). "A Test for Independence Based on the Correlation
  Dimension." *Econometric Reviews* 15(3), 197-235.
- Brock, W.A., Hsieh, D.A. & LeBaron, B. (1991). *Nonlinear
  Dynamics, Chaos, and Instability*. MIT Press.
- Kanzler, L. (1999). "Very Fast and Correctly Sized Estimation
  of the BDS Statistic." Oxford Dept. of Economics WP.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The BDS statistic tests iid-ness via the correlation integral:
for embedding dimension m and distance eps,

    C_m(eps) = (2 / (N(N-1))) * #{(i<j): ||X_i - X_j||_inf < eps}

Under iid, C_m = C_1^m and the standardized statistic
``w_m = sqrt(N) * (C_m - C_1^m) / sigma_m`` is asymptotically
N(0,1), where the Kanzler-style variance uses the standard
formula

    sigma_m^2 = 4 * [ K^m + 2*sum_{j=1}^{m-1} K^{m-j} C_1^{2j}
                    + (m-1)^2 C_1^{2m} - m^2 K C_1^{2m-2} ],

with ``K`` the probability two histories share a boundary
segment (third-order correlation integral
``P(||X_i-X_j||<eps and |x_{i+m}-x_{j+m}|<eps)`` estimated as
C_{m+1}-adjusted counts). We use the common practitioner form
with K from the pairs whose *last coordinate also matches*
(Kanzler 1999 / Brock et al. standard tabulation form), and
eps = 1.0 * sd(x). The synth compares iid innovations (null:
|w| in the central range) against a deterministic tent-map
chaotic path (dependence: reject) — the canonical BDS demo.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _corr_int(x: FloatArray, m: int, eps: float) -> float:
    """Correlation integral C_m(eps) via embedded pairs count."""
    n = x.size - m + 1
    if n < 20:
        raise ValueError("too short for embedding")
    emb = np.lib.stride_tricks.sliding_window_view(x, m)
    # count pairs with sup-norm < eps
    cnt = 0.0
    for i in range(n):
        d = np.abs(emb[i] - emb[i + 1 :]).max(axis=1)
        cnt += float(np.sum(d < eps))
    return cnt / (n * (n - 1) / 2.0)


def _corr_int_k(x: FloatArray, m: int, eps: float) -> float:
    """K(eps): triples correlation term for the variance form."""
    n = x.size - m
    if n < 20:
        raise ValueError("too short for embedding")
    emb = np.lib.stride_tricks.sliding_window_view(x, m + 1)
    cnt = 0.0
    for i in range(n):
        d = np.abs(emb[i] - emb[i + 1 :]).max(axis=1)
        cnt += float(np.sum(d < eps))
    return cnt / (n * (n - 1) / 2.0)


def bds_stat(x: FloatArray, m: int = 4, eps_frac: float = 1.0) -> dict[str, float]:
    """BDS w-statistic for embedding dimensions 2..m."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 200 or not np.all(np.isfinite(xx)):
        raise ValueError("bad series")
    if np.std(xx) < 1e-12:
        raise ValueError("degenerate series")
    xx = (xx - np.mean(xx)) / np.std(xx)
    eps = eps_frac  # standardized data -> eps = eps_frac * 1
    out: dict[str, float] = {}
    c1 = _corr_int(xx, 1, eps)
    if c1 <= 1e-9 or c1 >= 1.0 - 1e-9:
        raise ValueError("degenerate C1")
    n_eff = xx.size
    for mm in range(2, m + 1):
        cm = _corr_int(xx, mm, eps)
        k = _corr_int_k(xx, mm, eps)
        c1p = c1
        var = 4.0 * (
            k**mm
            + 2.0 * sum(k ** (mm - j) * c1p ** (2 * j) for j in range(1, mm))
            + (mm - 1) ** 2 * c1p ** (2 * mm)
            - mm**2 * k * c1p ** (2 * mm - 2)
        )
        if var <= 1e-12:
            raise ValueError("degenerate variance")
        w = math.sqrt(n_eff - mm) * (cm - c1p**mm) / math.sqrt(var)
        out[f"w{mm}"] = float(w)
        out[f"p{mm}"] = float(2.0 * _stats.norm.sf(abs(w)))
    return out


def _tent_map(n: int, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.uniform(0.01, 0.99)
    for t in range(1, n):
        x[t] = 2.0 * min(x[t - 1], 1.0 - x[t - 1])
    return np.asarray(x)


def synth_bds(
    seed: int = 20261231 + 321,
    n: int = 900,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC iid null vs deterministic tent-map path."""
    rng = np.random.default_rng(seed)
    iid = rng.normal(0.0, 1.0, n)
    chaos = _tent_map(n, seed + 1)
    return np.asarray(iid), np.asarray(chaos)


def bench_bds(seed: int = 20261231 + 321) -> dict[str, float]:
    """Wave-55 self-check: iid passes, tent map rejected."""
    iid, chaos = synth_bds(seed=seed)
    r_null = bds_stat(iid, m=3)
    r_chaos = bds_stat(chaos, m=3)
    ok = r_null["p2"] > 0.05 and r_chaos["p2"] < 1e-6
    ok = ok and r_chaos["p3"] < 1e-6
    return {
        "synthetic_w2_null": r_null["w2"],
        "synthetic_p2_null": r_null["p2"],
        "synthetic_w2_chaos": r_chaos["w2"],
        "synthetic_p2_chaos": r_chaos["p2"],
        "synthetic_p3_chaos": r_chaos["p3"],
        "synthetic_score": float(ok),
    }
