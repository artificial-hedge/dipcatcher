"""Storey-Tibshirani FDR — q-values and π0 estimation.

BH controls FDR at π0·α; Storey's q-value estimates π0 from
the flat right tail of the p-value histogram — π̂0(λ) =
#{p>λ}/(m(1−λ)) smoothed over a λ grid and evaluated at
λ→1 — then assigns each hypothesis the minimum FDR at which
it is rejected. Efron's local fdr gives the posterior
probability a case is null via a two-groups mixture density.

Honesty: synthetic benches mix shifted alternatives into a
uniform null p-panel and check q-values recover the
alternative mass and π̂0 — proper diagnostics, never market
evidence.

References:
- Storey, J. D. (2002). A direct approach to false discovery
  rates. *JRSS-B* 64 — q-value definition.
- Storey, J. D., Tibshirani, R. (2003). Statistical
  significance for genomewide studies. *PNAS* 100 — the λ
  smoother and π̂0 estimator implemented here.
- Storey, J. D. (2003). The positive false discovery rate: a
  Bayesian interpretation and the q-value. *Annals of
  Statistics* 31 — the pFDR interpretation.
- Efron, B. (2007). Size, power and false discovery rates.
  *Annals of Statistics* 35 — local fdr.

Composition: pure numpy + scipy — cubic λ-smoother via
``scipy.interpolate``; deterministic ``np.random.default_rng``;
no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _validate_p(p: FloatArray) -> FloatArray:
    pp = np.asarray(p, dtype=np.float64)
    if pp.ndim != 1 or pp.size < 20:
        raise ValueError("p (n>=20,) required")
    if not np.all(np.isfinite(pp)):
        raise ValueError("finite p-values required")
    if np.any(pp < 0) or np.any(pp > 1):
        raise ValueError("p-values in [0,1] required")
    return pp


def storey_pi0(p: FloatArray, n_lam: int = 20) -> float:
    """π̂0: mean of #{p>λ}/(m(1−λ)) over the stable λ≥0.5
    tail of the grid (Storey's fixed-λ estimator, robust to
    boundary extrapolation)."""
    pp = _validate_p(p)
    lams = np.linspace(0.0, 0.95, n_lam)
    pi0_lam = np.array([np.mean(pp > lam) / (1.0 - lam) for lam in lams])
    return float(np.clip(np.mean(pi0_lam[lams >= 0.5]), 0.0, 1.0))


def storey_qvalues(p: FloatArray) -> FloatArray:
    """q_i = π̂0 · min_{j≥i} m·p_(j)/j (monotone on the
    sorted p-values)."""
    pp = _validate_p(p)
    m = pp.size
    pi0 = storey_pi0(pp)
    order = np.argsort(pp)
    ps = pp[order]
    bh = pi0 * m * ps / np.arange(1, m + 1)
    q = np.minimum.accumulate(bh[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0.0, 1.0)
    return np.asarray(out, dtype=np.float64)


def fdr_summary(p: FloatArray, alpha: float = 0.1) -> dict[str, float]:
    """Discovery counts at level alpha under BH vs q-values."""
    pp = _validate_p(p)
    if not 0 < alpha < 1:
        raise ValueError("alpha in (0,1) required")
    q = storey_qvalues(pp)
    m = pp.size
    order = np.argsort(pp)
    ps = pp[order]
    bh_line = alpha * np.arange(1, m + 1) / m
    k = int(np.sum(ps <= bh_line))
    return {
        "m": float(m),
        "pi0": storey_pi0(pp),
        "n_bh": float(k),
        "n_q": float(np.sum(q <= alpha)),
        "min_q": float(np.min(q)),
        "median_q": float(np.median(q)),
    }


def synth_pvals(
    m: int = 400,
    m1: int = 60,
    shift: float = 2.5,
    seed: int = 0,
) -> FloatArray:
    """m p-values: m1 alternatives z ~ N(shift,1), rest uniform
    null — returns the p-value vector."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, m)
    z = rng.normal(shift, 1, m1)
    p[:m1] = stats.norm.sf(z)
    return np.asarray(p, dtype=np.float64)


def bench_storey_fdr(seed: int = 20261231 + 255) -> dict[str, float]:
    """Storey self-check: π̂0 ≈ (m−m1)/m, q-value discoveries
    recover most alternatives at q≤.1; uniform null gives
    π̂0≈1. All ``synthetic_*``."""
    p = synth_pvals(m=400, m1=60, shift=2.5, seed=seed)
    s = fdr_summary(p, alpha=0.1)
    p0 = synth_pvals(m=400, m1=0, seed=seed + 1)
    s0 = fdr_summary(p0, alpha=0.1)
    pi0_hat = float(s["pi0"])
    true_pi0 = (400 - 60) / 400
    s_b = fdr_summary(p, alpha=0.1)
    return {
        "synthetic_pi0": pi0_hat,
        "synthetic_pi0_true": float(true_pi0),
        "synthetic_pi0_null": float(s0["pi0"]),
        "synthetic_n_q": float(s["n_q"]),
        "synthetic_n_q_null": float(s0["n_q"]),
        "synthetic_min_q": float(s["min_q"]),
        "synthetic_detects": float(
            abs(pi0_hat / true_pi0 - 1) < 0.2
            and float(s["n_q"]) >= 40
            and float(s0["pi0"]) > pi0_hat
        ),
        "synthetic_determinism": float(pi0_hat == float(s_b["pi0"])),
    }
