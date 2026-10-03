"""Exponential-Hawkes calibration for the ZI-LOB event stream.

The market-order clock of ``zi_lob_simulator`` is Poisson within a regime;
this module fits the self-exciting alternative

    lambda*(t) = mu + sum_{t_i < t} alpha * exp(-beta (t - t_i))

by exact maximum likelihood and reports whether the data support any
branching at all. It is the calibration half of the Hawkes story — the
``impulse_mm`` lane supplies the simulator.

- ``hawkes_exp_loglik`` — closed-form log-likelihood via the standard
  O(n) recursion R_i = exp(-beta*dt_i)(1 + R_{i-1}), lambda_i = mu +
  alpha*R_i, plus the compensator term mu*T + (alpha/beta)*sum(1 -
  exp(-beta*(T - t_i))).
- ``fit_hawkes_exp`` — L-BFGS-B over log-params with restarts; the
  branching ratio n = alpha/beta is reported, and ``stationary=False``
  when the optimum sits at the n<1 boundary (honest flag, never clamped).
- ``hawkes_ogata`` — Ogata thinning sampler (test + bench ground truth).
- ``hawkes_cal_bench`` — sealed ``hawkes_cal.v1`` receipt: parameter
  recovery on synthetic Hawkes + a fit on the ZI-LOB MO stream (which
  should report near-zero branching — a real negative control).

SYNTHETIC only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

HAWKES_CAL_SCHEMA = "hawkes_cal.v1"


def _check_times(times: NDArray[np.float64]) -> NDArray[np.float64]:
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 3:
        raise ValueError("times must be a 1-D array with >= 3 events")
    if not np.all(np.isfinite(t)):
        raise ValueError("times must be finite")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if t[0] <= 0.0:
        raise ValueError("times must be positive")
    return t


def hawkes_exp_loglik(times: NDArray[np.float64], mu: float, alpha: float, beta: float) -> float:
    """Exact log-likelihood of an exponential Hawkes process.

    lambda*(t) = mu + alpha * sum_i exp(-beta (t - t_i)). Returns -inf for
    parameter vectors that leave a nonpositive conditional intensity —
    callers treat that as infeasible rather than an error.
    """
    t = _check_times(times)
    mu, alpha, beta = float(mu), float(alpha), float(beta)
    if mu <= 0.0 or alpha < 0.0 or beta <= 0.0:
        return float("-inf")
    horizon = float(t[-1])
    r = 0.0
    ll = 0.0
    prev = 0.0
    for ti in t:
        dt = ti - prev
        r = math.exp(-beta * dt) * (1.0 + r)
        lam = mu + alpha * r
        if lam <= 0.0:
            return float("-inf")
        ll += math.log(lam)
        prev = ti
    tail = t[-1] - prev  # noqa: F841 — kept for readability of compensator
    comp = mu * horizon + (alpha / beta) * float(np.sum(1.0 - np.exp(-beta * (horizon - t))))
    return ll - comp


@dataclass(frozen=True)
class HawkesFit:
    """Fitted exponential-Hawkes parameters."""

    mu: float
    alpha: float
    beta: float
    branching: float  # alpha / beta: stationary iff < 1
    stationary: bool
    loglik: float
    n_events: int


def _fit_once(
    t: NDArray[np.float64], x0: tuple[float, float, float]
) -> tuple[float, tuple[float, float, float]]:
    def nll(z: NDArray[np.float64]) -> float:
        mu, alpha, beta = np.exp(z)
        return -hawkes_exp_loglik(t, mu, alpha, beta)

    res = minimize(nll, np.log(np.asarray(x0)), method="L-BFGS-B")
    mu, alpha, beta = (float(v) for v in np.exp(res.x))
    return float(-res.fun), (mu, alpha, beta)


def fit_hawkes_exp(
    times: NDArray[np.float64],
    *,
    inits: tuple[tuple[float, float, float], ...] | None = None,
) -> HawkesFit:
    """MLE fit with log-parameter L-BFGS-B over a small deterministic grid.

    Restarts are deterministic (no rng). ``stationary`` is the honest flag:
    the fit reports whatever the likelihood prefers; a supercritical best
    fit is reported, not projected back inside n<1.
    """
    t = _check_times(times)
    span = float(t[-1] - t[0])
    rate0 = t.size / span
    starts = inits or (
        (rate0, 0.01 * rate0, 1.0),
        (0.5 * rate0, 0.5 * rate0, 2.0),
        (rate0, 0.9 * rate0, 10.0),
    )
    best_ll = float("-inf")
    best = (rate0, 0.0, 1.0)
    for x0 in starts:
        ll, params = _fit_once(t, x0)
        if ll > best_ll:
            best_ll, best = ll, params
    mu, alpha, beta = best
    branching = alpha / beta if beta > 0.0 else float("inf")
    return HawkesFit(
        mu=mu,
        alpha=alpha,
        beta=beta,
        branching=branching,
        stationary=bool(branching < 1.0),
        loglik=best_ll,
        n_events=int(t.size),
    )


def hawkes_ogata(
    mu: float, alpha: float, beta: float, horizon: float, *, seed: int
) -> NDArray[np.float64]:
    """Ogata thinning sampler for the exponential Hawkes process."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    mu, alpha, beta, horizon = (
        _pos_finite(mu, "mu"),
        _nonneg(alpha, "alpha"),
        _pos_finite(beta, "beta"),
        _pos_finite(horizon, "horizon"),
    )
    if alpha >= beta:
        raise ValueError("supercritical Hawkes (alpha>=beta) is not samplable")
    rng = np.random.default_rng(seed)
    times: list[float] = []
    t_now = 0.0
    while True:
        lam = mu + sum(alpha * math.exp(-beta * (t_now - ti)) for ti in times)
        dt = float(rng.exponential(1.0 / lam))
        t_next = t_now + dt
        if t_next >= horizon:
            break
        lam_next = mu + sum(alpha * math.exp(-beta * (t_next - ti)) for ti in times)
        if rng.random() * lam <= lam_next:
            times.append(t_next)
        t_now = t_next
    return np.asarray(times, dtype=float)


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


def _nonneg(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {x!r}")
    return v


def zi_mo_times(config: Any, horizon: float) -> NDArray[np.float64]:
    """Market-order arrival times from one ZI-LOB run."""
    from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator

    _pos_finite(horizon, "horizon")
    sim = ZILobSimulator(config)
    times: list[float] = []
    while sim.t < horizon:
        if sim.step() == "market":
            times.append(sim.t)
    return np.asarray(times, dtype=float)


def hawkes_cal_bench(
    *,
    horizon: float = 400.0,
    seed: int = 0,
    zi_horizon: float = 800.0,
) -> dict[str, Any]:
    """Sealed ``hawkes_cal.v1`` receipt: recovery + ZI negative control."""
    from quant_fund.microstructure.zi_lob_simulator import santa_fe_config

    _pos_finite(horizon, "horizon")
    _pos_finite(zi_horizon, "zi_horizon")

    # 1) recovery on a known Hawkes stream
    mu0, a0, b0 = 0.5, 0.8, 4.0
    truth = hawkes_ogata(mu0, a0, b0, horizon, seed=seed)
    fit_h = fit_hawkes_exp(truth)

    # 2) negative control: the ZI-LOB MO stream is regime-Poisson, so a
    #    Hawkes fit should find ~no self-excitation
    mo = zi_mo_times(santa_fe_config(seed=seed + 1), zi_horizon)
    fit_zi = fit_hawkes_exp(mo) if mo.size >= 3 else None

    payload: dict[str, Any] = {
        "schema": HAWKES_CAL_SCHEMA,
        "kind": "hawkes_cal",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "disclaimer": (
            "Exponential-Hawkes MLE on synthetic streams; the ZI-LOB fit "
            "is a negative control (Poisson-within-regime arrivals), "
            "never market evidence."
        ),
        "recovery": {
            "mu_true": mu0,
            "alpha_true": a0,
            "beta_true": b0,
            "mu_hat": float(fit_h.mu),
            "alpha_hat": float(fit_h.alpha),
            "beta_hat": float(fit_h.beta),
            "branching_hat": float(fit_h.branching),
            "mu_rel_err": float(abs(fit_h.mu - mu0) / mu0),
            "alpha_rel_err": float(abs(fit_h.alpha - a0) / a0),
            "n_events": fit_h.n_events,
            "stationary": fit_h.stationary,
        },
        "zi_negative_control": (
            {
                "n_mo_events": int(mo.size),
                "branching_hat": float(fit_zi.branching),
                "alpha_hat": float(fit_zi.alpha),
                "stationary": fit_zi.stationary,
            }
            if fit_zi is not None
            else {"n_mo_events": int(mo.size), "skipped": "too_few_events"}
        ),
    }
    payload["payload_sha256"] = hash_bytes(json.dumps(payload, sort_keys=True).encode())
    return payload


__all__ = [
    "HAWKES_CAL_SCHEMA",
    "HawkesFit",
    "fit_hawkes_exp",
    "hawkes_cal_bench",
    "hawkes_exp_loglik",
    "hawkes_ogata",
    "zi_mo_times",
]
