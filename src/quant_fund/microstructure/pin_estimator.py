"""Easley-O'Hara probability-of-informed-trading (PIN) estimation.

Each day nature draws a news event with probability ``alpha``; the
event is bad news with probability ``delta``. Uninformed buys and
sells arrive at rate ``eps``; on news days the informed side adds
``mu`` to buys (good news) or sells (bad news). The day's
buy/sell counts are therefore a three-component Poisson mixture::

    P(B, S) = (1-a) P(B;e) P(S;e)
            + a(1-d) P(B;e+m) P(S;e)
            + ad    P(B;e) P(S;e+m)

and ``PIN = a*m / (a*m + 2*e)``.

The naive likelihood underflows immediately — ``(1+m/e)^B`` overflows
for moderate ``B*m/e``. This module uses the Lin & Ke (2011) /
Easley-Lopez de Prado-O'Hara (2010) factorization, which reduces every
term to a sum inside a single ``log``::

    log P = -2e + (B+S) ln e - ln B! - ln S!
          + log( 1-a + a(1-d) e^{-m}(1+m/e)^B + ad e^{-m}(1+m/e)^S )

with ``e^{-m}(1+m/e)^B = exp(B ln(1+m/e) - m)`` evaluated in log space.

The MLE is fitted by L-BFGS-B from a multi-start grid (the objective
has multiple local optima — single-start PIN estimates are a known
failure mode). Inference is a seeded nonparametric bootstrap over
days, matching how the literature reports PIN uncertainty.

`simulate_pin_days` generates the exact model DGP, so the bench
reports parameter-recovery error against known truth. Evidence class:
SYNTHETIC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

PIN_SCHEMA = "pin_estimator.v1"


def _check_params(alpha: float, delta: float, mu: float, eps: float) -> None:
    for name, v, lo, hi in (
        ("alpha", alpha, 0.0, 1.0),
        ("delta", delta, 0.0, 1.0),
        ("mu", mu, 0.0, math.inf),
        ("eps", eps, 0.0, math.inf),
    ):
        if not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f"{name} must lie in [{lo}, {hi}], got {v!r}")
    if eps <= 0.0:
        raise ValueError("eps must be > 0 (degenerate day without uninformed flow)")


def log_likelihood_day(
    buys: float, sells: float, alpha: float, delta: float, mu: float, eps: float
) -> float:
    """Lin-Ke-factorized log-likelihood of one day's (B, S) counts."""
    _check_params(alpha, delta, mu, eps)
    if buys < 0 or sells < 0:
        raise ValueError("counts must be non-negative")
    ratio = 1.0 + mu / eps

    def _log(x: float) -> float:
        return math.log(x) if x > 0 else -math.inf

    log_terms = (
        _log(1.0 - alpha),
        _log(alpha) + _log(1.0 - delta) - mu + buys * math.log(ratio),
        _log(alpha) + _log(delta) - mu + sells * math.log(ratio),
    )
    mx = max(log_terms)
    if mx == -math.inf:
        return -math.inf
    mix = mx + math.log(sum(math.exp(t - mx) for t in log_terms if t > -math.inf))
    return (
        -2.0 * eps
        + (buys + sells) * math.log(eps)
        - math.lgamma(buys + 1)
        - math.lgamma(sells + 1)
        + mix
    )


def neg_log_likelihood(params: NDArray[np.float64], days: NDArray[np.float64]) -> float:
    alpha, delta, mu, eps = (float(p) for p in params)
    if eps <= 0 or not (0 <= alpha <= 1) or not (0 <= delta <= 1) or mu < 0:
        return 1e12
    total = 0.0
    for row in days:
        ll = log_likelihood_day(row[0], row[1], alpha, delta, mu, eps)
        if not math.isfinite(ll):
            return 1e12
        total += ll
    return -total


@dataclass(frozen=True)
class PinFit:
    alpha: float
    delta: float
    mu: float
    eps: float
    pin: float
    loglik: float
    n_days: int
    converged: bool


def pin_of(alpha: float, delta: float, mu: float, eps: float) -> float:
    denom = alpha * mu + 2.0 * eps
    return alpha * mu / denom if denom > 0 else 0.0


def fit_pin(days: NDArray[np.float64]) -> PinFit:
    """Multi-start L-BFGS-B MLE over (alpha, delta, mu, eps)."""
    days = np.asarray(days, dtype=float)
    if days.ndim != 2 or days.shape[1] != 2 or len(days) < 10:
        raise ValueError("days must be an (n>=10, 2) count matrix")
    if not np.all(np.isfinite(days)) or np.any(days < 0):
        raise ValueError("days must be finite non-negative counts")

    mean_b = float(np.mean(days[:, 0]))
    mean_s = float(np.mean(days[:, 1]))
    base = max(0.25 * (mean_b + mean_s), 1.0)
    best: PinFit | None = None
    for a0 in (0.1, 0.3, 0.5):
        for d0 in (0.2, 0.5, 0.8):
            x0 = np.asarray([a0, d0, base * 0.5, base], dtype=float)
            res = minimize(
                neg_log_likelihood,
                x0,
                args=(days,),
                method="L-BFGS-B",
                bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, None), (1e-9, None)],
                options={"maxiter": 500},
            )
            a, d, m, e = (float(v) for v in res.x)
            fit = PinFit(
                alpha=a,
                delta=d,
                mu=m,
                eps=e,
                pin=pin_of(a, d, m, e),
                loglik=-float(res.fun),
                n_days=len(days),
                converged=bool(res.success),
            )
            if best is None or fit.loglik > best.loglik:
                best = fit
    if not (best is not None):
        raise ValueError("best is not None")  # noqa: S101 - multi-start always returns a row
    return best


def pin_bootstrap_ci(
    days: NDArray[np.float64], *, n_boot: int = 200, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for PIN by resampling days."""
    if n_boot < 50:
        raise ValueError("n_boot must be >= 50")
    days = np.asarray(days, dtype=float)
    rng = np.random.default_rng(seed)
    pins: list[float] = []
    n = len(days)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            pins.append(fit_pin(days[idx]).pin)
        except ValueError:
            continue
    if len(pins) < n_boot // 2:
        raise ValueError("bootstrap collapsed: too few successful refits")
    lo, hi = np.quantile(np.asarray(pins), [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def simulate_pin_days(
    *,
    alpha: float,
    delta: float,
    mu: float,
    eps: float,
    n_days: int,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Exact PIN-model DGP: n_days of (buys, sells) counts."""
    _check_params(alpha, delta, mu, eps)
    if n_days < 1:
        raise ValueError("n_days must be >= 1")
    rng = np.random.default_rng(seed)
    news = rng.random(n_days) < alpha
    bad = rng.random(n_days) < delta
    lam_b = eps + np.where(news & ~bad, mu, 0.0)
    lam_s = eps + np.where(news & bad, mu, 0.0)
    buys = rng.poisson(lam_b)
    sells = rng.poisson(lam_s)
    return np.stack([buys, sells], axis=1).astype(float)


def pin_estimator_bench(
    *,
    n_days: int = 250,
    n_boot: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Recovery drill: fit PIN on simulated days with known truth.

    Two scenarios: informed flow present (alpha=0.35, delta=0.3,
    mu=80, eps=50) and pure ZI flow (alpha=0, PIN=0). The second is
    the important honesty check — a broken estimator *finds* informed
    trading in pure noise.
    """
    scenarios: dict[str, Any] = {}
    for name, truth in (
        ("informed", {"alpha": 0.35, "delta": 0.3, "mu": 80.0, "eps": 50.0}),
        ("pure_zi", {"alpha": 0.0, "delta": 0.5, "mu": 80.0, "eps": 50.0}),
    ):
        days = simulate_pin_days(n_days=n_days, seed=seed, **truth)
        fit = fit_pin(days)
        lo, hi = pin_bootstrap_ci(days, n_boot=n_boot, seed=seed)
        scenarios[name] = {
            "truth": truth | {"pin": pin_of(**truth)},
            "fit": {
                "alpha": fit.alpha,
                "delta": fit.delta,
                "mu": fit.mu,
                "eps": fit.eps,
                "pin": fit.pin,
                "loglik": fit.loglik,
                "converged": fit.converged,
            },
            "pin_ci95": [lo, hi],
            "pin_recovery_err": abs(fit.pin - pin_of(**truth)),
        }
    payload = {
        "schema": PIN_SCHEMA,
        "kind": "pin_estimator",
        "n_days": n_days,
        "n_boot": n_boot,
        "seed": seed,
        "scenarios": scenarios,
        "informed_recovery_ok": scenarios["informed"]["pin_recovery_err"] < 0.10,
        "pure_zi_pin_small": scenarios["pure_zi"]["fit"]["pin"] < 0.10,
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
