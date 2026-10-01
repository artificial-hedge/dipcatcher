"""Accelerated failure time — Weibull AFT with right censoring.

log T = xβ + σ·ε, ε standard smallest-extreme-value, gives Weibull
lifetimes with shape κ = 1/σ. Covariates rescale time multiplicatively
(survival acceleration), complementary to Cox's proportional hazards.
Full-information MLE on right-censored data:

  uncensored: log f(t) = -log σ - log t + z - exp(z),  z = (log t - xβ)/σ
  censored:   log S(t) = -exp(z)

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure β/σ recovery on generated
censored Weibull lifetimes — never market evidence.

References:
- Buckley, J., James, I. (1979). Linear regression with censored
  data. *Biometrika* 66, 429-436 — semiparametric AFT least
  squares (the semiparametric twin to this parametric fit).
- Wei, L. J. (1992). The accelerated failure time model: a useful
  alternative to the Cox regression model in survival analysis.
  *Statistics in Medicine* 11, 1871-1879.
- Kalbfleisch, J. D., Prentice, R. L. (2002). *The Statistical
  Analysis of Failure Time Data*, 2nd ed. — the Weibull AFT
  likelihood used.
- Lawless, J. F. (2003). *Statistical Models and Methods for
  Lifetime Data*, 2nd ed. — SEV link and censoring contributions.

Composition: pure numpy + scipy — L-BFGS-B MLE on (β, log σ),
observed-information SEs; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _check(
    t: FloatArray, d: FloatArray, x: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    tt = np.asarray(t, dtype=np.float64).ravel()
    dd = np.asarray(d, dtype=np.float64).ravel()
    xx = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if xx.shape[0] != tt.size:
        xx = xx.T
    n = tt.size
    if n < 40 or dd.size != n or xx.shape[0] != n:
        raise ValueError("t/d/x must align, n>=40")
    if not (np.all(np.isfinite(tt)) and np.all(np.isfinite(dd)) and np.all(np.isfinite(xx))):
        raise ValueError("finite inputs required")
    if np.any(tt <= 0):
        raise ValueError("lifetimes must be positive")
    if not np.all((dd == 0) | (dd == 1)):
        raise ValueError("event indicator must be 0/1")
    if dd.sum() < 10:
        raise ValueError("need >=10 observed events")
    if np.linalg.matrix_rank(xx) < xx.shape[1]:
        raise ValueError("x must have full column rank")
    return tt, dd, xx


def aft_fit(t: FloatArray, d: FloatArray, x: FloatArray) -> dict[str, float]:
    """Weibull AFT MLE: log T = xβ + σ·ε, right-censored d."""
    tt, dd, xx = _check(t, d, x)
    n = tt.size
    lt = np.log(tt)
    xx1 = np.column_stack([np.ones(n), xx])
    k = xx1.shape[1]

    def nll(theta: FloatArray) -> float:
        beta = theta[:k]
        sig = float(np.exp(theta[k]))
        z = (lt - xx1 @ beta) / sig
        ez = np.exp(np.clip(z, -30, 30))
        ll = np.where(
            dd > 0.5,
            -np.log(sig) - lt + z - ez,
            -ez,
        )
        return -float(np.sum(ll))

    # start: OLS on log-times of uncensored
    keep = dd > 0.5
    b0 = np.linalg.lstsq(xx1[keep], lt[keep], rcond=None)[0]
    s0 = float(np.std(lt[keep] - xx1[keep] @ b0))
    th0 = np.concatenate([b0, [np.log(max(s0, 0.1))]])
    res = minimize(
        nll,
        th0,
        method="L-BFGS-B",
        bounds=[(None, None)] * k + [(np.log(0.05), np.log(5.0))],
        options={"maxiter": 400},
    )
    th = res.x
    beta = th[:k]
    sig = float(np.exp(th[k]))

    # observed-information SEs on beta (numeric Hessian diag)
    eps_fd = 1e-5
    se = np.zeros(k)
    for j in range(k):
        tp = th.copy()
        tp[j] += eps_fd
        tm = th.copy()
        tm[j] -= eps_fd
        h = (nll(tp) - 2 * nll(th) + nll(tm)) / eps_fd**2
        se[j] = 1.0 / math.sqrt(max(h, 1e-12))

    # reference: naive OLS on log-times ignoring censoring
    b_naive = np.linalg.lstsq(xx1, lt, rcond=None)[0]
    # median survival at x=mean: exp(x̄β)·(ln2)^σ
    med = float(np.exp(xx.mean(0) @ beta[1:]) * (math.log(2)) ** sig)

    return {
        "n": float(n),
        "events": float(dd.sum()),
        "censor_share": float(1 - dd.mean()),
        "sigma": sig,
        "kappa": float(1 / sig),
        "median_at_mean": med,
        **{f"beta_{j}": float(v) for j, v in enumerate(beta)},
        **{f"se_{j}": float(v) for j, v in enumerate(se)},
        "beta_naive_1": float(b_naive[1]),
        "z_1": float(beta[1] / max(se[1], 1e-12)),
        "p_1": float(2 * (1 - norm.cdf(abs(float(beta[1] / max(se[1], 1e-12)))))),
    }


def synth_aft(
    n: int = 800,
    beta: float = -0.5,
    sigma: float = 0.6,
    censor_rate: float = 0.35,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """AFT DGP: log T = β·x + σ·ε (SEV), administrative right
    censoring drawn to target ``censor_rate``."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(0.0, 1.0, n)])
    eps = -np.log(-np.log(rng.uniform(1e-9, 1 - 1e-9, n)))  # SEV
    lt = beta * x[:, 0] + sigma * eps
    t = np.exp(lt)
    cen = np.exp(rng.uniform(0, 1, n) * (np.quantile(lt, 1 - censor_rate) + 0.5))
    d = (t <= cen).astype(np.float64)
    t_obs = np.minimum(t, cen)
    return {"t": t_obs, "d": d, "x": x, "beta_true": np.array([beta])}


def bench_aft_model(seed: int = 20261231 + 224) -> dict[str, float]:
    """AFT self-check: β and σ recovered under 35% censoring;
    naive OLS-on-log-time is biased toward zero.
    All ``synthetic_*``."""
    d = synth_aft(beta=-0.5, sigma=0.6, seed=seed)
    out = aft_fit(np.asarray(d["t"]), np.asarray(d["d"]), np.asarray(d["x"]))
    d0 = synth_aft(beta=0.0, seed=seed + 1)
    out0 = aft_fit(np.asarray(d0["t"]), np.asarray(d0["d"]), np.asarray(d0["x"]))
    out_b = aft_fit(np.asarray(d["t"]), np.asarray(d["d"]), np.asarray(d["x"]))

    b1 = float(out["beta_1"])
    b_naive = float(out["beta_naive_1"])
    return {
        "synthetic_beta1": b1,
        "synthetic_beta1_err": float(abs(b1 + 0.5)),
        "synthetic_sigma": float(out["sigma"]),
        "synthetic_sigma_err": float(abs(float(out["sigma"]) - 0.6)),
        "synthetic_naive_err": float(abs(b_naive + 0.5)),
        "synthetic_beats_naive": float(abs(b1 + 0.5) < abs(b_naive + 0.5)),
        "synthetic_p1": float(out["p_1"]),
        "synthetic_null_beta1": float(abs(out0["beta_1"])),
        "synthetic_detects": float(abs(b1 + 0.5) < 0.15 and float(out["p_1"]) < 0.05),
        "synthetic_determinism": float(b1 == float(out_b["beta_1"])),
    }
