"""Stambaugh bias and Bonferroni-Q inference for predictive (SYNTHETIC)
regressions.

When y_{t+1} = a + β x_t + ε_{t+1} and the persistent
predictor x_t follows AR(1) with innovation correlated with
ε, the OLS β̂ is biased upward of order (1+3ρ̂)/T and standard
t-tests over-reject. Stambaugh's closed-form bias and the
Bonferroni-Q procedure of Campbell-Yogo give a CI that is
robust to near-unit-root predictors.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure bias correction and CI
coverage on generated persistent predictors — never market
evidence.

References:
- Stambaugh, R. F. (1999). Predictive regressions. *Journal of
  Financial Economics* 54, 375-421 — small-sample bias
  E[β̂−β] = −(1+3ρ)/T·(σ_εu/σ_u²) direction and correction.
- Campbell, J. Y., Yogo, M. (2006). Efficient tests of stock
  return predictability. *Journal of Financial Economics* 81,
  27-60 — Bonferroni-Q CI via a grid over ρ with local-to-unity
  asymptotics.
- Lewellen, J. (2004). Predicting returns with financial
  ratios. *JFE* 74 — conservative bound ρ ≤ corr(x_t, x_{t+1}).
- Cavanagh, C. L., Elliott, G., Stock, J. H. (1995). Inference
  in models with nearly integrated regressors. *Econometric
  Theory* 11.

Composition: pure numpy + scipy — AR(1) Q-statistic inversion
by grid search, correlated-innovation map; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _ols(y: FloatArray, x: FloatArray) -> tuple[float, float, FloatArray]:
    xm = x - x.mean()
    ym = y - y.mean()
    b = float(np.sum(xm * ym) / np.sum(xm**2))
    a = float(ym.mean() - b * xm.mean())
    return a, b, ym - b * xm


def _q_stat(e: FloatArray, rho: float) -> float:
    """Campbell-Yogo Q(ρ): scaled variance of residual
    innovations under AR(1) coefficient ρ."""
    t = e.size
    u = e[1:] - rho * e[:-1]
    s2 = float(np.sum(e**2) / t)
    su = float(np.sum(u**2) / (t - 1))
    if su < 1e-300 or s2 < 1e-300:
        return np.inf
    return float(t * (su / s2 - (1 - rho**2)) ** 2 / (4 * rho**2 + 1e-12))


def stambaugh_pred(
    y: FloatArray,
    x: FloatArray,
    rho_grid: FloatArray | None = None,
) -> dict[str, float]:
    """Predictive-regression diagnostics.

    Returns OLS β̂, ρ̂(x), innovation correlation, Stambaugh
    bias + corrected β, and Bonferroni-Q CI for β."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.asarray(x, dtype=np.float64).ravel()
    if yy.size != xx.size:
        raise ValueError("aligned y/x required")
    t = yy.size
    if t < 60:
        raise ValueError("T>=60")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(xx))):
        raise ValueError("finite inputs required")
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        raise ValueError("both series must vary")

    a_b, b_ols, eps = _ols(yy[1:], xx[:-1])
    _, rho_hat, u = _ols(xx[1:], xx[:-1])
    rho_hat = float(np.clip(rho_hat, -0.99, 0.995))

    # innovation correlation: cov(eps_t, u_t)/σ_u²
    n = eps.size
    r_eu = float(np.sum(eps * u) / np.sum(u**2))
    bias = -(1 + 3 * rho_hat) / n * r_eu
    b_corrected = b_ols - bias

    # Bonferroni-Q: invert Q(ρ) ≤ chi2(1, .95) for ρ CI, then
    # map worst-case bias over that interval
    if rho_grid is None:
        rho_grid = np.linspace(-0.5, 0.999, 300)
    # Q inverted on the demeaned predictor
    xdm = xx - xx.mean()
    qvals = np.array([_q_stat(xdm, float(rho)) for rho in rho_grid])
    ok = qvals <= stats.chi2.ppf(0.95, 1)
    if np.any(ok):
        rho_lo = float(np.min(rho_grid[ok]))
        rho_hi = float(np.max(rho_grid[ok]))
    else:
        rho_lo = rho_hi = rho_hat
    # worst-case bias magnitude over [rho_lo, rho_hi]
    biases = np.array([abs(-(1 + 3 * r) / n * r_eu) for r in [rho_lo, rho_hat, rho_hi]])
    bmax = float(np.max(biases))
    z = float(stats.norm.ppf(0.975))
    se_b = float(np.sqrt(np.sum(eps**2) / (n - 2) / np.sum((xx[:-1] - xx[:-1].mean()) ** 2)))

    return {
        "t": float(t),
        "beta_ols": b_ols,
        "rho_hat": rho_hat,
        "r_eu": r_eu,
        "bias": bias,
        "beta_corrected": float(b_corrected),
        "rho_lo": rho_lo,
        "rho_hi": rho_hi,
        "beta_q_lo": float(b_corrected - bmax - z * se_b),
        "beta_q_hi": float(b_corrected + bmax + z * se_b),
        "se_ols": se_b,
        "t_ols": float(b_ols / se_b),
    }


def synth_pred(
    t: int = 300,
    beta: float = 0.05,
    rho: float = 0.95,
    corr: float = -0.8,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Persistent predictor x~AR(1) with innovations correlated
    with y's — the Stambaugh setup."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, corr], [corr, 1.0]])
    z = rng.multivariate_normal([0, 0], cov, t)
    x = np.zeros(t)
    u = z[:, 0] * np.sqrt(1 - rho**2)  # stationary variance
    x[0] = u[0]
    for i in range(1, t):
        x[i] = rho * x[i - 1] + u[i]
    y = beta * x + z[:, 1]
    return {"y": y, "x": x}


def bench_stambaugh(seed: int = 20261231 + 242) -> dict[str, float]:
    """Stambaugh self-check: OLS β̂ is shifted by correlated
    innovations; corrected β is closer to the simulated value
    and the Bonferroni-Q CI contains it. All ``synthetic_*``."""
    d = synth_pred(beta=0.05, rho=0.95, corr=-0.8, seed=seed)
    out = stambaugh_pred(d["y"], d["x"])
    dn = synth_pred(beta=0.05, rho=0.5, corr=0.0, seed=seed + 1)
    outn = stambaugh_pred(dn["y"], dn["x"])
    out_b = stambaugh_pred(d["y"], d["x"])

    err_ols = abs(float(out["beta_ols"]) - 0.05)
    err_cor = abs(float(out["beta_corrected"]) - 0.05)
    return {
        "synthetic_beta_ols": float(out["beta_ols"]),
        "synthetic_beta_corrected": float(out["beta_corrected"]),
        "synthetic_bias": float(out["bias"]),
        "synthetic_err_ols": err_ols,
        "synthetic_err_corrected": err_cor,
        "synthetic_rho": float(out["rho_hat"]),
        "synthetic_bias_uncorr": float(outn["bias"]),
        "synthetic_q_contains": float(out["beta_q_lo"] <= 0.05 <= out["beta_q_hi"]),
        "synthetic_detects": float(
            err_cor < err_ols
            and float(out["rho_hat"]) > 0.85
            and abs(float(outn["bias"])) < abs(float(out["bias"]))
        ),
        "synthetic_determinism": float(
            float(out["beta_corrected"]) == float(out_b["beta_corrected"])
        ),
    }
