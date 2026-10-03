"""Beck-Katz (1995) panel-corrected standard errors and
Parks (1967) contemporaneous-correlation GLS.

Canonical references:

- Beck & Katz (1995) 'What to do (and not to do) with
  time-series cross-section data' APSR 89 — PCSE: OLS
  point estimates with a sandwich covariance whose meat
  is the T x T block Kronecker of the cross-sectional
  residual covariance:
      Var(b) = (X'X)^-1 [ sum_t x_t' Sigma x_t ] (X'X)^-1
  where Sigma_ij = <e_i e_j>_t over the sample period.
- Parks (1967) 'Efficient estimation of a system of
  regression equations when disturbances are both
  serially and cross-sectionally correlated' JASA 62 —
  feasible GLS using the same Sigma estimator. (FGLS is
  known to be anti-conservative when T ~ N — hence the
  PCSE recommendation; we expose both.)

Balanced panel assumed: N units x T periods, y ~ Xb with
unit-clustered, cross-sectionally correlated residuals.

`bench_pcse` builds a 10-unit x 40-period panel with
strongly cross-correlated residuals (equicorrelation
0.5) and gates: PCSE standard errors larger than the
heteroskedasticity-only Eicker-Huber-White ones (the
missed correlation inflates the true variance), and
coverage of the true beta better than naive OLS SEs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_panel(
    y: FloatArray, X: FloatArray, n_units: int, n_periods: int
) -> tuple[FloatArray, FloatArray]:
    ya = np.asarray(y, dtype=np.float64).ravel()
    xa = np.asarray(X, dtype=np.float64)
    nt = n_units * n_periods
    if n_units < 2 or n_periods < 4:
        raise ValueError("panel too small")
    if ya.size != nt or xa.shape[0] != nt or xa.ndim != 2:
        raise ValueError("shape mismatch (stack by unit, unit-major)")
    if not np.isfinite(ya).all() or not np.isfinite(xa).all():
        raise ValueError("non-finite")
    return ya, xa


def _ols(y: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta, y - x @ beta


def beck_katz(
    y: FloatArray,
    X: FloatArray,
    n_units: int,
    n_periods: int,
) -> dict[str, object]:
    """OLS point estimates with panel-corrected SEs.

    Panels stacked unit-major: rows [unit0 t0..tT, unit1 ...].
    """
    ya, xa = _check_panel(y, X, n_units, n_periods)
    k = xa.shape[1]
    beta, resid = _ols(ya, xa)
    e = resid.reshape(n_units, n_periods)  # unit x time
    sigma = (e @ e.T) / n_periods  # N x N contemporaneous
    xtx_inv = np.linalg.inv(xa.T @ xa)
    meat = np.zeros((k, k))
    for t in range(n_periods):
        xt = xa[t::n_periods]  # (N, k): rows t, T+t, 2T+t...
        meat += xt.T @ sigma @ xt
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    # EHW (heteroskedasticity-only) reference SE
    ehw = xtx_inv @ (xa.T * (resid**2) @ xa) @ xtx_inv
    return {
        "beta": beta,
        "se_pcse": se,
        "se_ehw": np.sqrt(np.clip(np.diag(ehw), 0, None)),
        "sigma": sigma,
        "resid": resid,
    }


def parks_fgls(
    y: FloatArray,
    X: FloatArray,
    n_units: int,
    n_periods: int,
) -> dict[str, object]:
    """Parks (1967) feasible GLS with contemporaneous
    correlation + unit AR(1) correction skipped (the
    contemporaneous part is the identified one)."""
    ya, xa = _check_panel(y, X, n_units, n_periods)
    bk = beck_katz(ya, xa, n_units, n_periods)
    sigma = np.asarray(bk["sigma"])
    # GLS transform: y* = C^-1 y, X* = C^-1 X per period,
    # where C C^T = Sigma
    chol = np.linalg.cholesky(sigma + 1e-10 * np.eye(n_units))
    cinv = np.linalg.inv(chol)
    ys = np.zeros_like(ya)
    xs = np.zeros_like(xa)
    for t in range(n_periods):
        idx = np.arange(t, n_units * n_periods, n_periods)
        ys[idx] = cinv @ ya[idx]
        xs[idx] = cinv @ xa[idx]
    beta, resid = _ols(ys, xs)
    cov = np.linalg.inv(xs.T @ xs)
    return {
        "beta": beta,
        "se": np.sqrt(np.clip(np.diag(cov), 0, None)),
        "sigma": sigma,
        "resid": resid,
    }


def bench_pcse(seed: int = 519) -> dict[str, float]:
    """SYNTHETIC: equicorrelated (rho=.5) residuals; PCSE SEs
    must exceed EHW and give correct coverage over a
    replicate study."""
    n_u, n_t, n_rep = 10, 40, 60
    rng = np.random.default_rng(seed)
    cover_pcse = 0
    cover_ehw = 0
    ratios = []
    sq_err = 0.0
    b_true = np.array([0.5, 1.5])
    for _ in range(n_rep):
        X = np.column_stack([np.ones(n_u * n_t), rng.normal(0, 1, n_u * n_t)])
        # equicorrelated unit residuals
        sig_e = 0.5 * np.ones((n_u, n_u)) + 0.5 * np.eye(n_u)
        L = np.linalg.cholesky(sig_e)
        e = (L @ rng.normal(0, 1.0, (n_u, n_t))).ravel()
        y = X @ b_true + e
        out = beck_katz(y, X, n_u, n_t)
        se_p = np.asarray(out["se_pcse"])
        se_e = np.asarray(out["se_ehw"])
        b = np.asarray(out["beta"])
        z_p = np.abs(b - b_true) / se_p
        z_e = np.abs(b - b_true) / se_e
        cover_pcse += int((z_p < 1.96).all())
        cover_ehw += int((z_e < 1.96).all())
        ratios.append(float((se_p / np.maximum(se_e, 1e-12)).mean()))
        sq_err += float((b[1] - b_true[1]) ** 2)
    cov_pc = cover_pcse / n_rep
    cov_eh = cover_ehw / n_rep
    mean_ratio = float(np.mean(ratios))
    if mean_ratio <= 1.0:
        raise ValueError("PCSE not wider than EHW")
    if cov_pc <= cov_eh:
        raise ValueError("PCSE coverage not better")
    return {
        "synthetic_pcse_ehw_ratio": mean_ratio,
        "synthetic_coverage_pcse": cov_pc,
        "synthetic_coverage_ehw": cov_eh,
        "synthetic_beta1_rmse": float(np.sqrt(sq_err / n_rep)),
    }
