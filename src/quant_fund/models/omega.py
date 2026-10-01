"""McDonald's omega — composite reliability from a one-factor
solution.

McDonald (1999): given standardized loadings lambda_j of a single-
factor model fit to the item correlation matrix, composite
reliability is

    omega = (sum_j lambda_j)^2 / ((sum_j lambda_j)^2 + sum_j (1 - lambda_j^2))

Unlike coefficient alpha (tau-equivalence lower bound), omega
admits congeneric items — it equals reliability exactly when the
one-factor model holds. Loadings here come from principal-axis
factoring of the tetrachoric-free Pearson correlation (one
iteration of communality refinement, mirroring the factor
machinery already in the repo but self-contained).

Honesty: omega can understate reliability when the factor model
fits poorly — we emit the implied residual off-diagonal RMS so
callers can check fit. The bench uses a planted congeneric item
pool (omega recovers the true composite reliability) and
independent items (omega ~ 0). Fail-closed on <3 items or a
degenerate correlation.

References: McDonald (1999) "Test Theory: A Unified Treatment"
ch. 6; Zinbarg, Revelle, Yovel & Li (2005) "Cronbach's alpha,
Revelle's beta, and McDonald's omega"; Bentler (2009).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_items(x: FloatArray) -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 10 or a.shape[1] < 3:
        raise ValueError("bad item matrix")
    if not np.isfinite(a).all():
        raise ValueError("non-finite input")
    return a


def _one_factor_loadings(corr: FloatArray) -> tuple[FloatArray, float]:
    """Principal-axis single factor with one communality pass."""
    p = corr.shape[0]
    red = corr.copy()
    np.fill_diagonal(red, 1.0 - 1.0 / np.diag(np.linalg.pinv(corr)))
    vals, vecs = np.linalg.eigh(red)
    top = int(np.argmax(vals))
    if vals[top] <= 0:
        raise ValueError("no positive factor root")
    lam = vecs[:, top] * np.sqrt(vals[top])
    resid = corr - np.outer(lam, lam)
    np.fill_diagonal(resid, 0.0)
    rms = float(np.sqrt((resid**2).sum() / (p * (p - 1))))
    return np.asarray(lam, dtype=np.float64), rms


def omega_total(x: FloatArray) -> dict[str, float]:
    """McDonald's omega (total) for an (n, k) item matrix."""
    a = _check_items(x)
    corr = np.asarray(np.corrcoef(a, rowvar=False), dtype=np.float64)
    if not np.isfinite(corr).all():
        raise ValueError("degenerate correlation")
    lam, rms = _one_factor_loadings(corr)
    sl = float(lam.sum())
    uniques = float((1.0 - lam**2).sum())
    omega = sl * sl / (sl * sl + uniques)
    return {
        "omega": float(omega),
        "resid_rms": rms,
        "sum_loading": sl,
        "mean_loading2": float((lam**2).mean()),
    }


def bench_omega(seed: int = 20261231 + 442) -> dict[str, float]:
    """SYNTHETIC check — congeneric pool recovers composite reliability."""
    rng = np.random.default_rng(seed)
    n, k = 500, 8
    theta = rng.standard_normal(n)
    loading = np.linspace(0.4, 1.0, k)
    x = (
        theta[:, None] * loading[None, :]
        + rng.standard_normal((n, k)) * np.sqrt(1 - loading**2)[None, :]
    )
    out = omega_total(x)
    noise = omega_total(rng.standard_normal((n, k)))
    # true composite reliability: Var(sum lam theta + err) share of signal
    true_omega = float(loading.sum() ** 2 / (loading.sum() ** 2 + (1 - loading**2).sum()))
    if abs(out["omega"] - true_omega) > 0.15 or noise["omega"] > 0.35:
        raise ValueError(
            f"omega off: hat={out['omega']:.3f} true={true_omega:.3f} noise={noise['omega']:.3f}"
        )
    return {
        "synthetic_omega": out["omega"],
        "synthetic_omega_true": true_omega,
        "synthetic_omega_noise": noise["omega"],
        "synthetic_omega_resid_rms": out["resid_rms"],
        "score": 1.0,
    }
