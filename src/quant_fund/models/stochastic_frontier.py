"""Stochastic frontier analysis — normal/half-normal composed error (SYNTHETIC).

Production frontier: y_i = x_i'β + v_i - u_i, with v ~ N(0, σ_v²)
noise and u ~ half-N(0, σ_u²) one-sided inefficiency. Parameters by
MLE on the composed-error likelihood (Aigner-Lovell-Schmidt);
firm-level efficiency via the Jondrow conditional mean
E[u | v - u] = μ_* + σ_*·φ(μ_*/σ_*)/Φ(μ_*/σ_*).

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure frontier-slope recovery and
efficiency-ranking fidelity on generated panels — never market
evidence.

References:
- Aigner, Lovell, Schmidt (1977). Formulation and estimation of
  stochastic frontier production function models. *J. Econometrics* 6.
- Jondrow, Lovell, Materov, Schmidt (1982). On the estimation of
  technical inefficiency. *J. Econometrics* 19.
- Kumbhakar, Lovell (2000). *Stochastic Frontier Analysis*, ch. 3-4.
- Battese, Coelli (1988). Prediction of firm-level technical
  efficiencies. *J. Econometrics* 38.

Composition: pure numpy/scipy — composite loglik MLE, Jondrow
conditional mean; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _as2(m: FloatArray, name: str) -> FloatArray:
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 2-D matrix required")
    return a


def _as1(v: FloatArray, name: str) -> FloatArray:
    a = np.asarray(v, dtype=np.float64).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 1-D vector required")
    return a


def sfa_fit(
    y: FloatArray,
    x: FloatArray,
) -> dict[str, float | FloatArray]:
    """Normal/half-normal SFA MLE.

    Composed error ε = v - u, v~N(0,σv²), u~half-N(0,σu²):
      ll = Σ [-ln σ + ln Φ(-ε_i·λ/σ) + ln φ(ε_i/σ)],
      σ² = σv² + σu², λ = σu/σv.
    Parameterize (β, log σv, log σu) and optimize by BFGS from an
    OLS start."""
    ya = _as1(y, "y")
    xa = _as2(x, "x")
    n = ya.size
    k = xa.shape[1]
    if xa.shape[0] != n:
        raise ValueError("y/x length mismatch")
    if n < k + 8:
        raise ValueError("n too small")

    Xd = np.column_stack([np.ones(n), xa])
    b0, *_ = np.linalg.lstsq(Xd, ya, rcond=None)
    resid0 = ya - Xd @ b0
    s0 = max(float(resid0.std()), 1e-3)

    def nll(theta: FloatArray) -> float:
        beta = theta[: k + 1]
        sv = math.exp(float(theta[k + 1]))
        su = math.exp(float(theta[k + 2]))
        s2 = sv * sv + su * su
        s_ = math.sqrt(s2)
        lam = su / sv
        eps = ya - Xd @ beta
        a_ = -eps * lam / s_
        # loglik = sum[ -ln s + ln Φ(a) + ln φ(ε/s) ]
        ll = float(
            -n * math.log(s_)
            + np.log(np.maximum(norm.cdf(a_), 1e-300)).sum()
            + norm.logpdf(eps / s_).sum()
        )
        return -ll if math.isfinite(ll) else 1e18

    theta0 = np.concatenate([b0, [math.log(s0), math.log(s0 * 0.5)]])
    res = minimize(nll, theta0, method="BFGS", options={"maxiter": 400})
    th = res.x
    beta = th[: k + 1]
    sv = math.exp(float(th[k + 1]))
    su = math.exp(float(th[k + 2]))
    lam = su / sv
    s_ = math.sqrt(sv * sv + su * su)
    eps = ya - Xd @ beta

    # Jondrow conditional mean E[u|ε]: μ_* = -ε·σu²/σ², σ_* = σvσu/σ
    mu_star = -eps * su * su / (s_ * s_)
    sig_star = sv * su / s_
    cond = mu_star + sig_star * norm.pdf(mu_star / sig_star) / np.maximum(
        norm.cdf(mu_star / sig_star), 1e-12
    )
    eff = np.exp(-cond)  # technical efficiency ∈ (0,1]

    return {
        "beta": np.asarray(beta),
        "sigma_v": sv,
        "sigma_u": su,
        "lambda": lam,
        "eps": eps,
        "efficiency": eff,
        "efficiency_mean": float(eff.mean()),
        "efficiency_min": float(eff.min()),
        "loglik": -float(res.fun),
        "converged": float(res.success),
        "n_obs": float(n),
    }


def synth_sfa(
    n: int = 400,
    beta: float = 0.7,
    sigma_u: float = 0.6,
    sigma_v: float = 0.3,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Frontier panel: y = β·x + v − u, u~half-normal."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    v = rng.normal(0.0, sigma_v, n)
    u = np.abs(rng.normal(0.0, sigma_u, n))
    y = beta * x + v - u
    return {
        "y": y,
        "x": x.reshape(-1, 1),
        "u_true": u,
        "beta_true": np.array([beta]),
    }


def bench_stochastic_frontier(seed: int = 20261231 + 207) -> dict[str, float]:
    """SFA self-check: β recovered, σu detected vs σv, efficiency
    ranking tracks true u, converged flag honest. All ``synthetic_*``."""
    d = synth_sfa(seed=seed, sigma_u=0.6, sigma_v=0.3)
    out = sfa_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    beta_hat = float(np.asarray(out["beta"])[1])
    su_hat = float(out["sigma_u"])

    # OLS benchmark can't separate σu — its resid skew is the test
    y = np.asarray(d["y"])
    x = np.asarray(d["x"])
    Xd = np.column_stack([np.ones(y.size), x])
    bo, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    r_ols = y - Xd @ bo
    skew_ols = float(
        ((r_ols - r_ols.mean()) ** 3).mean() / max(r_ols.std() ** 3, 1e-12)
    )  # composed error skews NEGATIVE (inefficiency subtracts)

    d0 = synth_sfa(seed=seed + 1, sigma_u=0.05, sigma_v=0.5)
    out0 = sfa_fit(np.asarray(d0["y"]), np.asarray(d0["x"]))
    su0 = float(out0["sigma_u"])

    # efficiency ranking fidelity: corr(E[u|ε], true u)
    eff_corr = float(np.corrcoef(np.asarray(out["efficiency"]), -np.asarray(d["u_true"]))[0, 1])
    out_b = sfa_fit(np.asarray(d["y"]), np.asarray(d["x"]))

    return {
        "synthetic_beta": beta_hat,
        "synthetic_beta_err": float(abs(beta_hat - 0.7)),
        "synthetic_sigma_u_hat": su_hat,
        "synthetic_sigma_u_detected": float(su_hat > 0.2),
        "synthetic_sigma_u_null": su0,
        "synthetic_null_sigma_u_small": float(su0 < su_hat * 0.6),
        "synthetic_ols_resid_skew": skew_ols,
        "synthetic_skew_negative": float(skew_ols < -0.3),
        "synthetic_eff_corr": eff_corr,
        "synthetic_detects": float(abs(beta_hat - 0.7) < 0.15 and su_hat > 0.2 and eff_corr > 0.5),
        "synthetic_determinism": float(
            float(np.asarray(out["beta"])[1]) == float(np.asarray(out_b["beta"])[1])
        ),
    }
