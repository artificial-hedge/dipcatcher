"""State-dependent local projections — Auerbach & Gorodnichenko.

Impulse responses whose coefficients switch smoothly between two
regimes (expansion/recession, boom/bust) via a logistic transition
on a state variable z_t (e.g. a cyclical gap):

    y_{t+h} = (1 - z_t) [α_E + β_E x_t + controls]
              + z_t [α_R + β_R x_t + controls] + eps

with z_t = F((s_t - c)/γ), F the logistic CDF. Per horizon h the
regression is OLS on interacted regressors; the estimated β_E(h),
β_R(h) trace state-dependent impulse response functions. Inference
is per-horizon OLS with HAC (Newey-West) standard errors.

References
----------
- Auerbach, A.J., Gorodnichenko, Y. (2012). "Measuring the output
  responses to fiscal policy." *AEJ: Economic Policy* 4(2).
- Auerbach, A.J., Gorodnichenko, Y. (2013). "Fiscal multipliers in
  recession and expansion." *NBER WP* w17447.
- Ramey, V.A., Zubairy, S. (2018). "Government spending
  multipliers in good times and in bad." *JPE* 126(2).
- Jordà, Ò. (2005). "Estimation and inference of impulse
  responses by local projections." *AER* 95(1) — the LP base.

Honesty
-------
SYNTHETIC DGP only; bench verifies the state split recovers the
imposed regime-dependent multipliers — not a macro claim.

Composition
-----------
Called by ``quant_fund.research.benches_w67.bench_state_dep_lp``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def transition_prob(s: FloatArray, c: float = 0.0, gamma: float = 1.5) -> FloatArray:
    """Logistic transition F((s-c)/gamma) ∈ (0,1); z→1 in bust."""
    s = np.asarray(s, dtype=float)
    if not (gamma > 0):
        raise ValueError("gamma must be positive")
    x = np.clip((s - c) / gamma, -60.0, 60.0)
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def _nw_se(x: FloatArray, e: FloatArray, lags: int) -> FloatArray:
    """Newey-West standard errors for OLS coefficients."""
    n = x.shape[0]
    xtwx_inv = np.linalg.inv(x.T @ x)
    s = np.zeros((x.shape[1], x.shape[1]))
    for lag in range(lags + 1):
        wt = 1.0 - lag / (lags + 1)
        g = (x[lag:] * e[lag:, None]).T @ (x[: n - lag] * e[: n - lag, None])
        s += wt * g if lag == 0 else wt * (g + g.T)
    cov = xtwx_inv @ s @ xtwx_inv * (n / (n - x.shape[1]))
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def state_lp(
    y: FloatArray,
    x: FloatArray,
    s: FloatArray,
    horizons: FloatArray | list[int],
    c: float = 0.0,
    gamma: float = 1.5,
    nw_lags: int = 4,
) -> dict[str, FloatArray]:
    """State-dependent LP impulse responses.

    y (n,) outcome; x (n,) impulse/shock; s (n,) transition state.
    Returns dict of arrays over horizons: beta_E (expansion,
    z->0), beta_R (recession, z->1), plus their Newey-West SEs.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    n = y.size
    if x.size != n or s.size != n or n < 40:
        raise ValueError("bad inputs")
    z = transition_prob(s, c, gamma)
    beta_e, beta_r, se_e, se_r = [], [], [], []
    for h in horizons:
        h = int(h)
        yy = y[h:]
        xx0 = x[: n - h]
        zz = z[: n - h]
        # regressors: [x*(1-z), x*z] (+ intercepts per state)
        xmat = np.column_stack([xx0 * (1 - zz), xx0 * zz, 1 - zz, zz])
        b = np.linalg.lstsq(xmat, yy, rcond=None)[0]
        e = yy - xmat @ b
        se = _nw_se(xmat, e, nw_lags)
        beta_e.append(float(b[0]))
        beta_r.append(float(b[1]))
        se_e.append(float(se[0]))
        se_r.append(float(se[1]))
    return {
        "beta_expansion": np.asarray(beta_e),
        "beta_recession": np.asarray(beta_r),
        "se_expansion": np.asarray(se_e),
        "se_recession": np.asarray(se_r),
    }


def bench_state_dependent_lp(seed: int = 20261231 + 393) -> dict[str, float]:
    """SYNTHETIC check — expansion/recession multipliers separated."""
    rng = np.random.default_rng(seed)
    n = 1200
    s = rng.standard_normal(n)
    x = rng.standard_normal(n)
    z = transition_prob(s, 0.0, 1.0)
    beta_e_true, beta_r_true = 1.0, 2.8
    # outcome: state-dependent response + AR-ish noise
    y = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        y[t] = 0.3 * y[t - 1] + (1 - z[t]) * beta_e_true * x[t] + z[t] * beta_r_true * x[t] + e[t]
    out = state_lp(y, x, s, horizons=[0, 1, 2], c=0.0, gamma=1.5)
    be0 = float(out["beta_expansion"][0])
    br0 = float(out["beta_recession"][0])
    if abs(be0 - beta_e_true) > 0.4 or abs(br0 - beta_r_true) > 0.5:
        raise ValueError(f"state betas off: {be0} {br0}")
    if not (br0 > be0 + 1.0):
        raise ValueError("recession multiplier not larger")
    # se ordering: recession SEs exist and finite
    if not np.all(np.isfinite(out["se_recession"])):
        raise ValueError("non-finite SEs")
    # impulse decays with horizon on this DGP
    if not (out["beta_recession"][0] > out["beta_recession"][2]):
        raise ValueError("horizon decay missing")
    return {
        "synthetic_sdlp_beta_expansion": be0,
        "synthetic_sdlp_beta_recession": br0,
        "synthetic_sdlp_gap": br0 - be0,
        "score": 1.0,
    }
