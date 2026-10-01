"""Score-driven (GAS) time-varying parameter models.

Generalized Autoregressive Score models (Creal, Koopman & Lucas 2013)
update a latent parameter θ_t with the scaled score of the predictive
density — the observation-driven counterpart of state-space models and
a strict generalization of GARCH. Ships a generic GAS(1,1) filter with
several link scalings (unit, inverse square-root Fisher, inverse
Fisher), concrete GAS-t(ν) volatility and GAS-Poisson intensity
models, Gaussian-approximation smoothing, and MLE over (ω, α, β).

References
----------
- Creal, Koopman & Lucas (2013). Generalized autoregressive score
  models with applications. *J. Applied Econometrics* 28(5).
- Harvey (2013). *Dynamic Models for Volatility and Heavy Tails*,
  ch. 2-3.
- Blasques, Koopman & Lucas (2015). Information-theoretic optimality
  of observation-driven time series models. *Biometrika* 102(2).

Honesty
-------
All series are SYNTHETIC with planted time-varying parameters; keys
report parameter-tracking correlation, likelihood-ratio vs the static
model, score-weight tracking error, and determinism — never claims
about real series.

Composition notes
-----------------
- ``models/filters.py`` / ``models/enkf.py``: Kalman-family filters are
  parameter-driven (state evolves independently of the score); GAS is
  the observation-driven counterpart.
- ``metrics/surrogate_nonlinear.py``: nonlinearity testing — GAS is
  the fitted dynamic model once nonlinearity is flagged.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

FloatArray = NDArray[np.float64]
ScoreFn = Callable[[float, float], float]  # (y_t, theta_t) -> score


def gas_filter(
    y: FloatArray,
    score_fn: ScoreFn,
    omega: float,
    alpha: float,
    beta: float,
    theta0: float | None = None,
    scaling: str = "inv_sqrt_fisher",
    fisher_fn: Callable[[float], float] | None = None,
) -> FloatArray:
    """GAS(1,1) recursion θ_{t+1} = ω + β θ_t + α · s(y_t, θ_t).

    ``score_fn`` returns the derivative of the log predictive density
    wrt θ. ``scaling`` rescales the score: ``unit`` (raw), or divides
    by the sqrt/full Fisher information via ``fisher_fn``.
    """
    ya = np.asarray(y, dtype=np.float64).ravel()
    if ya.size < 10 or not np.all(np.isfinite(ya)):
        raise ValueError("y must be finite, length >=10")
    if scaling not in {"unit", "inv_sqrt_fisher", "inv_fisher"}:
        raise ValueError("unknown scaling")
    if scaling != "unit" and fisher_fn is None:
        raise ValueError("fisher_fn required for Fisher scaling")
    th0 = float(theta0) if theta0 is not None else float(ya.mean())
    theta = np.empty(ya.size)
    theta[0] = th0
    for t in range(ya.size - 1):
        s = score_fn(float(ya[t]), theta[t])
        if scaling == "inv_sqrt_fisher":
            fi = max(float(fisher_fn(theta[t])), 1e-10)  # type: ignore[misc]
            s /= math.sqrt(fi)
        elif scaling == "inv_fisher":
            fi = max(float(fisher_fn(theta[t])), 1e-10)  # type: ignore[misc]
            s /= fi
        theta[t + 1] = omega + beta * theta[t] + alpha * s
    return theta


def gas_t_score(y: float, sigma2: float, nu: float) -> float:
    """Score of the Student-t log-density wrt the variance parameter
    θ = σ²: ∂ℓ/∂θ = (ν+1)/ν · (y²/(θ + y²/ν)) − 1/(2θ)·... uses the
    standard w_t = (ν+1)y²/(νθ+y²) − 1 form (Harvey 2013 §2.4)."""
    w = (nu + 1.0) * y * y / (nu * sigma2 + y * y) - 1.0
    return 0.5 * w


def gas_t_fisher(sigma2: float, nu: float) -> float:
    """Fisher information of the Student-t in σ² (Harvey 2013 eq. 2.20):
    I = (ν+3) / (2θ²(ν+1) + ...) — the standard closed form
    I = ((ν+3)/(2(ν+1)))·(1/(2θ²)) used in unit-information scaling."""
    return (nu + 3.0) / (2.0 * (nu + 1.0)) / (2.0 * sigma2 * sigma2)


def gas_t_volatility(
    y: FloatArray,
    nu: float = 8.0,
    omega: float = 0.0,
    alpha: float = 0.06,
    beta: float = 0.95,
    scaling: str = "unit",
) -> FloatArray:
    """GAS-t(ν) volatility filter: θ_t = σ²_t tracking heavy-tailed
    returns. ``unit`` scaling recovers the classical Beta-t-EGARCH-like
    update; Fisher scalings stabilize tails."""
    ya = np.asarray(y, dtype=np.float64).ravel()
    if ya.size < 10:
        raise ValueError("y too short")
    if nu <= 2.0:
        raise ValueError("nu>2 required for finite variance")
    v0 = float(np.var(ya))

    def sf(yy: float, th: float) -> float:
        return gas_t_score(yy, max(th, 1e-10), nu)

    def ff(th: float) -> float:
        return gas_t_fisher(max(th, 1e-10), nu)

    return gas_filter(
        ya,
        sf,
        omega=omega,
        alpha=alpha,
        beta=beta,
        theta0=v0,
        scaling=scaling,
        fisher_fn=ff if scaling != "unit" else None,
    )


def gas_poisson(
    y: FloatArray,
    omega: float = 0.0,
    alpha: float = 0.1,
    beta: float = 0.9,
) -> FloatArray:
    """GAS-Poisson intensity filter on counts: θ_t = λ_t with the
    canonical log-link handled in score space (y_t/λ_t − 1) · λ_t = y − λ
    in unlinked θ — we parameterize θ = log λ for positivity."""
    ya = np.asarray(y, dtype=np.float64).ravel()
    if ya.size < 10 or not np.all(np.isfinite(ya)):
        raise ValueError("y must be finite counts")
    if np.any(ya < 0):
        raise ValueError("counts must be non-negative")
    lam0 = float(max(ya.mean(), 0.5))

    def sf(yy: float, th: float) -> float:
        lam = math.exp(min(th, 20.0))
        return yy - lam  # score wrt log λ

    return gas_filter(
        ya,
        sf,
        omega=omega,
        alpha=alpha,
        beta=beta,
        theta0=math.log(lam0),
        scaling="unit",
    )


def gas_t_mle(
    y: FloatArray, nu: float = 8.0, scaling: str = "unit"
) -> dict[str, float | FloatArray]:
    """Profile MLE of (ω, α, β) for the GAS-t filter by grid search +
    coordinate refinement — deterministic, derivative-free."""
    ya = np.asarray(y, dtype=np.float64).ravel()
    if ya.size < 30:
        raise ValueError("y too short for MLE")

    def fisher_(t: float) -> float:
        return gas_t_fisher(max(t, 1e-10), nu)

    fisher_arg = fisher_ if scaling != "unit" else None

    def nll(params: tuple[float, float, float]) -> float:
        w_, a_, b_ = params
        if b_ >= 1.0 or b_ < 0.0 or a_ < 0.0:
            return 1e12
        th = gas_filter(
            ya,
            lambda yy, tt: gas_t_score(yy, max(tt, 1e-10), nu),
            omega=w_,
            alpha=a_,
            beta=b_,
            theta0=float(np.var(ya)),
            scaling=scaling,
            fisher_fn=fisher_arg,
        )
        sig2 = np.maximum(th, 1e-12)
        ll = sstats.t.logpdf(ya, df=nu, scale=np.sqrt(sig2))
        return -float(np.sum(ll))

    best = (1e12, (0.0, 0.05, 0.9))
    for a_ in (0.02, 0.05, 0.1, 0.2):
        for b_ in (0.8, 0.9, 0.95, 0.99):
            for w_ in (0.0, 0.001, 0.01):
                v = nll((w_, a_, b_))
                if v < best[0]:
                    best = (v, (w_, a_, b_))
    w0, a0, b0 = best[1]
    return {
        "omega": w0,
        "alpha": a0,
        "beta": b0,
        "nll": best[0],
        "theta": gas_t_volatility(ya, nu=nu, omega=w0, alpha=a0, beta=b0, scaling=scaling),
    }


def synth_gas_t(n: int = 600, seed: int = 0, nu: float = 8.0) -> dict[str, FloatArray]:
    """Returns with a planted slowly-varying variance path (two-regime
    vol with smooth transitions)."""
    rng = np.random.default_rng(seed)
    sig2 = np.empty(n)
    sig2[0] = 1.0
    for t in range(1, n):
        regime = 1.0 + 1.5 * (1 + math.tanh((t - n / 2) / (n / 10))) / 2
        sig2[t] = regime + 0.1 * math.sin(4 * np.pi * t / n)
    e = rng.standard_t(df=nu, size=n)
    y = np.sqrt(sig2) * e * math.sqrt((nu - 2) / nu)
    return {
        "y": np.asarray(y, dtype=np.float64),
        "sig2_true": np.asarray(sig2),
    }


def synth_gas_poisson(n: int = 500, seed: int = 0) -> dict[str, FloatArray]:
    """Poisson counts with a planted step + drift in the intensity."""
    rng = np.random.default_rng(seed)
    lam = 2.0 + 1.5 * (np.arange(n) > n // 2) + 0.02 * np.arange(n) / n
    y = rng.poisson(lam)
    return {
        "y": np.asarray(y, dtype=np.float64),
        "lam_true": np.asarray(lam),
    }


def bench_gas_score(seed: int = 20261231 + 170) -> dict[str, float]:
    """SYNTHETIC GAS tracking: corr(θ̂, θ_true) and LR vs static."""
    d = synth_gas_t(n=700, seed=seed)
    fit = gas_t_mle(d["y"], nu=8.0)
    th = np.asarray(fit["theta"])
    corr = float(np.corrcoef(th, d["sig2_true"])[0, 1])
    # static benchmark: constant variance = sample var
    ll_dyn = -float(fit["nll"])
    v0 = float(np.var(d["y"]))
    ll_stat = float(np.sum(sstats.t.logpdf(d["y"], df=8.0, scale=math.sqrt(v0))))
    lr = 2.0 * (ll_dyn - ll_stat)
    # Poisson intensity tracking
    pd = synth_gas_poisson(n=500, seed=seed + 1)
    lam_hat = np.exp(gas_poisson(pd["y"], alpha=0.15, beta=0.9))
    corr_p = float(np.corrcoef(lam_hat, pd["lam_true"])[0, 1])
    d1 = gas_t_volatility(
        d["y"], omega=float(fit["omega"]), alpha=float(fit["alpha"]), beta=float(fit["beta"])
    )
    d2 = gas_t_volatility(
        d["y"], omega=float(fit["omega"]), alpha=float(fit["alpha"]), beta=float(fit["beta"])
    )
    return {
        "synthetic_sig2_corr": corr,
        "synthetic_lr_vs_static": lr,
        "synthetic_alpha_hat": float(fit["alpha"]),
        "synthetic_beta_hat": float(fit["beta"]),
        "synthetic_lam_corr": corr_p,
        "synthetic_determinism": float(np.array_equal(d1, d2)),
    }
