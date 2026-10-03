"""Fay-Herriot small-area estimation — area-level EBLUP.

Fay & Herriot (1979): for direct survey estimates y_i with known
sampling variances D_i and area-level covariates x_i, the basic
area-level model is

    y_i = x_i^T beta + v_i + e_i,   v_i ~ N(0, A), e_i ~ N(0, D_i)

The DL/ML estimate of A (Prasad-Rao / Fay-Herriot moment
estimator) is

    A_hat = max(0, [sum_i (y_i - x_i^T b_hat)^2 - sum_i (D_i - h_ii)]/(m - p))

with h_ii the leverage under GLS weights. The EBLUP shrinks the
direct estimate toward the synthetic predictor:
    theta_i = B_i y_i + (1 - B_i) x_i^T beta,  B_i = D_i/(A + D_i)

Honesty: A is estimated by the Fay-Herriot moment form (iterative
DL refinement inside); when A_hat = 0 the EBLUP collapses to the
synthetic estimator — both behaviors reported. The bench plants a
genuine random area effect (EBLUP beats direct and synthetic).
Fail-closed on negative variances or degenerate X.

References: Fay & Herriot (1979) "Estimates of income for small
places", JASA 74:269; Prasad & Rao (1990) MSE; Datta, Lahiri &
Maiti (2002); Rao & Molina (2015) "Small Area Estimation" 2nd ed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(
    y: FloatArray, x: FloatArray, d: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    a = np.asarray(y, dtype=float)
    m = np.asarray(x, dtype=float)
    v = np.asarray(d, dtype=float)
    if a.ndim != 1 or m.ndim != 2 or m.shape[0] != a.size or v.shape != a.shape:
        raise ValueError("bad inputs")
    if (
        a.size < 5
        or not np.isfinite(a).all()
        or not np.isfinite(m).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("non-finite input")
    if (v <= 0).any():
        raise ValueError("non-positive sampling variance")
    return a, m, v


def _gls_beta(x: FloatArray, y: FloatArray, w: FloatArray) -> FloatArray:
    xtwx = (x.T * w) @ x
    if np.linalg.matrix_rank(xtwx) < x.shape[1]:
        raise ValueError("singular design")
    return np.linalg.solve(xtwx, (x.T * w) @ y)


def _fh_moment_a(y: FloatArray, x: FloatArray, d: FloatArray, n_iter: int = 80) -> float:
    """Iterative FH moment estimator of A — Newton on
    F(A) = sum_i w_i r_i^2 - (m - p) = 0, w_i = 1/(A + D_i),
    with GLS refit and a step cap for stability."""
    m = y.size
    p = x.shape[1]
    a_var = float(max(np.median(d), 1e-6))
    cap = 5.0 * float(np.median(d))
    for _ in range(n_iter):
        w = 1.0 / (a_var + d)
        beta = _gls_beta(x, y, w)
        resid = y - x @ beta
        s = float((w * resid * resid).sum())
        deriv = float((w * w * resid * resid).sum())
        if deriv <= 0:
            break
        step = np.clip((s - (m - p)) / deriv, -cap, cap)
        a_new = max(0.0, a_var + float(step))
        if abs(a_new - a_var) < 1e-8 * (1 + a_var):
            a_var = a_new
            break
        a_var = a_new
    return float(a_var)


def fay_herriot(y: FloatArray, x: FloatArray, d: FloatArray) -> dict[str, float | FloatArray]:
    """EBLUP small-area estimates.

    ``y`` direct estimates, ``x`` (m, p) area covariates, ``d``
    known sampling variances.
    """
    a, m_x, v = _check(y, x, d)
    a_var = _fh_moment_a(a, m_x, v)
    w = 1.0 / (a_var + v)
    beta = _gls_beta(m_x, a, w)
    b_shrink = v / (a_var + v)
    theta = b_shrink * a + (1.0 - b_shrink) * (m_x @ beta)
    # g1 + g2 MSE approximation (Prasad-Rao leading terms, averaged
    # over areas since g1 is elementwise)
    g1 = v * a_var / (a_var + v)
    mse_est = float(
        g1.mean()
        + 2.0
        * float((((v / (a_var + v)) ** 2).mean()) * a_var**2)
        * float((1.0 / (a_var + v)).sum())
        / a.size**2
    )
    return {
        "a_var": a_var,
        "theta": np.asarray(theta, dtype=np.float64),
        "shrinkage": np.asarray(b_shrink, dtype=np.float64),
        "mse_est": max(mse_est, 0.0),
        "mean_shrinkage": float(b_shrink.mean()),
    }


def bench_fay_herriot(seed: int = 20261231 + 447) -> dict[str, float]:
    """SYNTHETIC check — EBLUP improves on direct + synthetic."""
    rng = np.random.default_rng(seed)
    m = 40
    x = np.stack([np.ones(m), rng.standard_normal(m)], axis=1)
    beta = np.array([2.0, 1.0])
    a_true = 0.8
    v = rng.uniform(0.3, 1.2, m)
    v_i = np.sqrt(a_true) * rng.standard_normal(m)  # area effects
    y = x @ beta + v_i + np.sqrt(v) * rng.standard_normal(m)
    out = fay_herriot(y, x, v)
    theta = np.asarray(out["theta"], dtype=float)
    truth = x @ beta + v_i
    eblup_mse = float(((theta - truth) ** 2).mean())
    direct_mse = float(((y - truth) ** 2).mean())
    synth_mse = float(((x @ beta - truth) ** 2).mean())
    a_hat = float(out["a_var"])
    if eblup_mse >= min(direct_mse, synth_mse) or abs(a_hat - a_true) > 0.6:
        raise ValueError(
            f"fh off: eblup={eblup_mse:.3f} direct={direct_mse:.3f} "
            f"synth={synth_mse:.3f} a_hat={a_hat:.3f}"
        )
    return {
        "synthetic_fh_a_hat": a_hat,
        "synthetic_fh_eblup_mse": eblup_mse,
        "synthetic_fh_direct_mse": direct_mse,
        "synthetic_fh_synth_mse": synth_mse,
        "score": 1.0,
    }
