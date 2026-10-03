"""Proxy SVAR + sign-restricted set identification.

Two complementary identification schemes for structural vector
autoregressions:

1. Proxy/instrumented SVAR (Stock & Watson 2012; Mertens & Ravn 2013):
   an external instrument ``z_t`` correlated with the target structural
   shock and uncorrelated with the others identifies one column of the
   impact matrix via the covariance of residuals with the instrument.
2. Sign restrictions (Uhlig 2005; Arias, Rubio-Ramírez & Waggoner 2018):
   random orthogonal rotations of the Cholesky factor, kept when the
   implied impulse responses satisfy a sign table — yielding a *set* of
   admissible models rather than a point estimate.

References
----------
- Stock & Watson (2012). Disentangling the channels of the 2007-09
  recession. *Brookings Papers on Economic Activity* 2012(1).
- Mertens & Ravn (2013). The dynamic effects of personal and corporate
  income tax changes. *American Economic Review* 103(4) — replication
  code instruments.
- Uhlig (2005). What are the effects of monetary policy on output?
  *Journal of Monetary Economics* 52(2).
- Arias, Rubio-Ramírez & Waggoner (2018). Inference based on SVARs
  identified with sign and zero restrictions. *Econometrica* 86(2).

Honesty
-------
All validation is SYNTHETIC: a bivariate VAR with a planted demand/
supply sign pattern and a generated instrument correlated only with
shock 1. Keys report instrument strength, sign compliance and IRF-shape
recovery on simulated data — never macro-economic conclusions.

Composition notes
-----------------
- ``models/var_coint.py``: reduced-form VAR fit/IRF/FEVD/spillover —
  this module consumes its ``var_fit`` output and adds identification.
- ``models/ms_var.py``: Markov-switching VAR — different regime model.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ArrayDict = dict[str, FloatArray | np.float64]


def _as_panel(y: FloatArray) -> FloatArray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] < 20 or y.shape[1] < 2:
        raise ValueError("y must be (T>=20, n>=2)")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")
    return y


def var_fit(y: FloatArray, p: int = 1) -> dict[str, FloatArray]:
    """Reduced-form VAR(p): OLS with intercept. Returns A (n x n p),
    Sigma_u, residuals U (T-p x n)."""
    y = _as_panel(y)
    if p < 1 or y.shape[0] - p < y.shape[1] * p + 4:
        raise ValueError("p too large for T")
    t, n = y.shape
    yy = y[p:]
    xx = np.column_stack([y[p - lag - 1 : t - lag - 1] for lag in range(p)] + [np.ones(t - p)])
    b, _res, _r, _s = np.linalg.lstsq(xx, yy, rcond=None)
    a = b[:-1].T.reshape(n, n * p)
    u = yy - xx @ b
    sig = u.T @ u / (t - p - n * p - 1)
    return {"A": a, "Sigma": np.asarray(sig), "U": np.asarray(u), "B": np.asarray(b)}


def companion_irf(a: FloatArray, n: int, horizon: int = 10) -> FloatArray:
    """MA(∞) coefficient matrices Ψ_h for h=0..horizon from A (n x n p)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != n or a.shape[1] % n:
        raise ValueError("A must be (n, n*p)")
    p = a.shape[1] // n
    psi = np.zeros((horizon + 1, n, n))
    psi[0] = np.eye(n)
    for h in range(1, horizon + 1):
        for j in range(1, min(h, p) + 1):
            psi[h] += a[:, (j - 1) * n : j * n] @ psi[h - j]
    return psi


def proxy_svar(
    y: FloatArray,
    z: FloatArray,
    p: int = 1,
    target_var: int = 0,
    horizon: int = 10,
) -> ArrayDict:
    """Instrumented SVAR identification of one shock.

    ``z`` (length T or T-p aligned to the residual sample) is an
    instrument for shock ``target_var``: Cov(z, u_target) != 0,
    Cov(z, u_others) ≈ 0. The identified column of the impact matrix B0
    is proportional to E[z u]; responses are scaled so shock 1 has a
    unit impact on the target variable (normalisation of Mertens-Ravn).

    Returns: irf (horizon+1, n), impact column b0_col, first-stage F,
    relevance stats.
    """
    y = _as_panel(y)
    z = np.asarray(z, dtype=np.float64).ravel()
    n = y.shape[1]
    if not (0 <= target_var < n):
        raise ValueError("target_var out of range")
    fit = var_fit(y, p)
    u = fit["U"]
    t_u = u.shape[0]
    if z.size == t_u + p:
        z = z[p:]
    if z.size != t_u:
        raise ValueError("z must align with the residual sample (T-p)")
    if not np.all(np.isfinite(z)):
        raise ValueError("z must be finite")
    # first stage: u_target ~ z
    zx = np.column_stack([z, np.ones(t_u)])
    beta_z, _r, _rk, _sv = np.linalg.lstsq(zx, u[:, target_var], rcond=None)
    u_hat = zx @ beta_z
    ssr = float(np.sum((u[:, target_var] - u_hat) ** 2))
    sst = float(np.sum((u[:, target_var] - u[:, target_var].mean()) ** 2))
    f_first = ((sst - ssr) / 1.0) / (ssr / max(t_u - 2, 1)) if ssr > 0 else float("inf")
    # relevance/exclusion diagnostics
    corrz = np.array([abs(np.corrcoef(z, u[:, j])[0, 1]) for j in range(n)])
    # impact vector proportional to Cov(z, u), unit-normalised on target
    cov_zu = np.array([float(np.cov(z, u[:, j], ddof=1)[0, 1]) for j in range(n)])
    scale = cov_zu[target_var]
    if abs(scale) < 1e-12:
        raise ValueError("instrument not relevant for target shock")
    b0_col = cov_zu / scale
    psi = companion_irf(fit["A"], n, horizon)
    irf = np.einsum("hij,j->hi", psi, b0_col)
    return {
        "irf": np.asarray(irf, dtype=np.float64),
        "b0_col": np.asarray(b0_col, dtype=np.float64),
        "first_stage_f": np.float64(f_first),
        "weak_instrument": np.float64(f_first < 10.0),
        "corr_z_target": np.float64(corrz[target_var]),
        "corr_z_max_other": np.float64(np.max(np.delete(corrz, target_var))),
    }


def _random_rotation(rng: np.random.Generator, n: int) -> FloatArray:
    """Uniform Haar orthogonal matrix via QR of a Gaussian."""
    m = rng.standard_normal((n, n))
    q, r = np.linalg.qr(m)
    d = np.sign(np.diag(r))
    q = q * d
    return np.asarray(q, dtype=np.float64)


def sign_restrict_draws(
    y: FloatArray,
    p: int = 1,
    restrictions: dict[tuple[int, int], int] | None = None,
    n_draws: int = 200,
    horizon: int = 10,
    seed: int = 0,
) -> ArrayDict:
    """Sign-restricted SVAR via Uhlig-style QR rotations.

    ``restrictions`` maps (response_var, horizon_index) -> +1/-1
    requiring the response of shock 0 (column 0 of the rotation) to
    variable ``response_var`` at horizon ``horizon_index`` to carry that
    sign. Draws are kept when ALL restrictions hold.

    Returns: kept rotation draws' IRFs (n_kept, horizon+1, n), keep rate,
    median and 16/84 percentile response bands.
    """
    y = _as_panel(y)
    if restrictions is None or not restrictions:
        raise ValueError("at least one sign restriction required")
    n = y.shape[1]
    for (v, h), sgn in restrictions.items():
        if not (0 <= v < n and 0 <= h <= horizon and sgn in (1, -1)):
            raise ValueError("invalid restriction key or sign")
    if n_draws < 1:
        raise ValueError("n_draws >= 1")
    fit = var_fit(y, p)
    chol = np.linalg.cholesky(fit["Sigma"])
    psi = companion_irf(fit["A"], n, horizon)
    rng = np.random.default_rng(seed)
    kept: list[FloatArray] = []
    for _ in range(n_draws):
        q = _random_rotation(rng, n)
        b0 = chol @ q
        col0 = b0[:, 0]
        resp = np.einsum("hij,j->hi", psi, col0)
        ok = all(
            math.copysign(1.0, resp[h, v] + 1e-300) == sgn for (v, h), sgn in restrictions.items()
        )
        if ok:
            kept.append(resp)
    arr = np.stack(kept) if kept else np.zeros((0, horizon + 1, n))
    med = np.median(arr, axis=0) if kept else np.full((horizon + 1, n), np.nan)
    lo = np.percentile(arr, 16, axis=0) if kept else np.full((horizon + 1, n), np.nan)
    hi = np.percentile(arr, 84, axis=0) if kept else np.full((horizon + 1, n), np.nan)
    return {
        "irfs": np.asarray(arr, dtype=np.float64),
        "n_kept": np.float64(len(kept)),
        "keep_rate": np.float64(len(kept) / n_draws),
        "median": np.asarray(med, dtype=np.float64),
        "lo16": np.asarray(lo, dtype=np.float64),
        "hi84": np.asarray(hi, dtype=np.float64),
    }


def synth_svar(
    t: int = 400,
    seed: int = 0,
    rho: float = 0.6,
) -> dict[str, FloatArray]:
    """Bivariate supply/demand VAR with a valid instrument for shock 1.

    Structural shocks eps ~ N(0, I); impact matrix B0 = [[1, 0.5],
    [-0.7, 1]] — shock 1 raises y0 and lowers y1 (demand-like), shock 2
    raises both (supply-like). Instrument z = 1.2*eps1 + 0.6*noise is
    correlated with shock 1 only.
    """
    if t < 100:
        raise ValueError("t >= 100")
    rng = np.random.default_rng(seed)
    b0 = np.array([[1.0, 0.5], [-0.7, 1.0]])
    a = np.array([[rho, 0.1], [0.0, 0.4]])
    eps = rng.standard_normal((t, 2))
    u = eps @ b0.T
    y = np.zeros((t, 2))
    for i in range(1, t):
        y[i] = a @ y[i - 1] + u[i]
    z = 1.2 * eps[:, 0] + 0.6 * rng.standard_normal(t)
    return {
        "y": np.asarray(y),
        "z": np.asarray(z),  # instrument aligned to T
        "eps": np.asarray(eps),
        "B0": np.asarray(b0),
        "A": np.asarray(a),
    }


def bench_proxy_svar(seed: int = 20261231 + 162) -> dict[str, float]:
    """SYNTHETIC proxy-SVAR + sign-restriction validation."""
    sim = synth_svar(t=600, seed=seed, rho=0.6)
    y = np.asarray(sim["y"], dtype=np.float64)
    z = np.asarray(sim["z"], dtype=np.float64)
    b0 = np.asarray(sim["B0"], dtype=np.float64)
    out = proxy_svar(y, z, p=1, target_var=0, horizon=8)
    # true IRF of shock 1 (unit on var 0)
    psi = companion_irf(var_fit(y, 1)["A"], 2, 8)
    true_col = b0[:, 0] / b0[0, 0]
    irf_true = np.einsum("hij,j->hi", psi, true_col)
    irf_est = out["irf"]
    err = float(np.linalg.norm(irf_est - irf_true) / np.linalg.norm(irf_true))
    # sign restrictions: shock lowers y1 at h=0 (demand pattern)
    rest = sign_restrict_draws(
        y,
        p=1,
        restrictions={(1, 0): -1, (0, 0): 1},
        n_draws=400,
        horizon=6,
        seed=seed,
    )
    # compliance: fraction of kept draws (should be > 0 and < 1 — set id)
    kept_rate = float(rest["keep_rate"])
    # weak instrument under noise instrument
    rng = np.random.default_rng(seed)
    noise_z: FloatArray = rng.standard_normal(int(y.shape[0]))
    weak = 0.0
    try:
        out2 = proxy_svar(y, noise_z, p=1, target_var=0, horizon=4)
        weak = float(out2["weak_instrument"])
    except ValueError:
        weak = 1.0
    # determinism
    d1 = proxy_svar(y, z, p=1, target_var=0, horizon=4)["irf"]
    d2 = proxy_svar(y, z, p=1, target_var=0, horizon=4)["irf"]
    return {
        "synthetic_irf_relerr": err,
        "synthetic_first_stage_f": float(out["first_stage_f"]),
        "synthetic_corr_z_target": float(out["corr_z_target"]),
        "synthetic_corr_z_max_other": float(out["corr_z_max_other"]),
        "synthetic_weak_instrument_flagged": weak,
        "synthetic_sign_keep_rate": kept_rate,
        "synthetic_sign_n_kept": float(rest["n_kept"]),
        "synthetic_determinism": float(np.array_equal(d1, d2)),
    }
