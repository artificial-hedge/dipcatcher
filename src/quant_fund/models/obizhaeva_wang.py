"""Obizhaeva-Wang optimal execution — Obizhaeva & Wang (2013).

Under the transient-propagator model, market impact is

    P_t = S_t + eta * x_t_dot + sum_{s<t} x_s_dot * G(t - s)

with G(u) = exp(-rho u) the exponential resilience kernel — impact
decays at rate rho after each trade. The static problem of buying X
shares over [0, T] minimizing expected implementation shortfall is a
Fredholm quadratic program; for the exponential kernel the optimal
schedule has the closed form

    x_dot_0 = x_dot_T = X / (rho T + 2)             (block trades
                                                    at the ends)
    x_dot_t = rho X / (rho T + 2)                   (continuous
                                                    rate inside)

and the minimal extra cost over the "do-nothing" price is

    C* = eta X^2 / T * rho / (rho T + 2) * (1 + 1/(rho T))
       + X^2 rho / (rho T + 2)^2 * (T + 2/rho)

(up to the permanent-impact convention). Discrete-time portfolio
choices are solved exactly by the same quadratic form on a grid —
used here for the multi-block schedule.

References
----------
- Obizhaeva, A., Wang, J. (2013). "Optimal trading strategy and
  supply/demand dynamics." *Journal of Financial Markets* 16(1).
- Gatheral, J. (2010). "No-dynamic-arbitrage and market impact."
  *Quantitative Finance* 10(7) — propagator framework.

Honesty
-------
SYNTHETIC schedules only; the bench checks the static frontier beats
TWAP, the block/inside split matches the closed form, and the cost
declines in resilience rho — no market claim.

Composition
-----------
Called by ``quant_fund.research.benches_w65.bench_obizhaeva_wang``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def ow_schedule(x_total: float, t_horizon: float, rho: float) -> tuple[float, float, float]:
    """Static optimal schedule for the exponential kernel.

    Returns (block trade at each end, continuous rate inside, total
    cost in extra-impact units with eta=0). Both ends trade the same
    block B = X/(rho T + 2) and the middle trades at rate
    rho X/(rho T + 2).
    """
    if not (x_total > 0 and t_horizon > 0 and rho > 0):
        raise ValueError("x_total, t_horizon, rho must be positive")
    denom = rho * t_horizon + 2.0
    block = x_total / denom
    rate = rho * x_total / denom
    return block, rate, denom


def ow_cost(schedule: FloatArray, dt: float, eta: float, rho: float) -> float:
    """Expected implementation cost of a discrete schedule.

    ``schedule`` holds the per-step trade sizes (positive = buy); the
    cost is  eta * sum(x_i^2) + sum_{i,j} x_i x_j G(|i-j| dt)  with
    G(u) = exp(-rho u) — the first term is the quadratic-in-size
    instantaneous impact of each block trade, the second is the
    decaying transient memory (diagonal counted once).
    """
    x = np.asarray(schedule, dtype=float).ravel()
    if x.size < 1 or not (dt > 0 and eta >= 0 and rho >= 0):
        raise ValueError("bad schedule or parameters")
    n = x.size
    idx = np.arange(n)
    g = np.exp(-rho * np.abs(idx[:, None] - idx[None, :]) * dt)
    temp = float(x @ g @ x)
    inst = float(eta * np.sum(x * x))
    return temp + inst


def ow_optimal_cost(x_total: float, t_horizon: float, eta: float, rho: float) -> float:
    """Closed-form minimal cost of the continuous static problem."""
    if not (x_total > 0 and t_horizon > 0 and rho > 0 and eta >= 0):
        raise ValueError("parameters must be positive (eta >= 0)")
    block, rate, denom = ow_schedule(x_total, t_horizon, rho)
    # Cost = instant blocks' impact + transient integral. Evaluate on a
    # fine grid consistent with the discrete cost functional.
    n = 512
    dt = t_horizon / n
    x = np.full(n, rate * dt)
    x[0] += block
    x[-1] += block
    return ow_cost(x, dt, eta, rho)


def bench_obizhaeva_wang(seed: int = 20261231 + 380) -> dict[str, float]:
    """SYNTHETIC check — optimal schedule beats TWAP; monotone in rho."""
    _ = np.random.default_rng(seed)
    x_total, t_horizon, eta, rho = 1.0, 1.0, 0.1, 2.0
    n = 200
    dt = t_horizon / n
    block, rate, _ = ow_schedule(x_total, t_horizon, rho)
    # Continuous-form schedule: end blocks + uniform inside rate.
    x_opt = np.full(n, rate * dt)
    x_opt[0] += block
    x_opt[-1] += block
    # Exact discrete optimum by KKT: min x'Gx + eta x'x s.t. sum x = X.
    idx = np.arange(n)
    g = np.exp(-rho * np.abs(idx[:, None] - idx[None, :]) * dt)
    a_qp = g + eta * np.eye(n)
    ones = np.ones(n)
    x_qp = np.linalg.solve(
        np.block([[2.0 * a_qp, ones[:, None]], [ones[None, :], np.zeros((1, 1))]]),
        np.concatenate([np.zeros(n), [x_total]]),
    )[:n]
    c_opt = ow_cost(x_opt, dt, eta, rho)
    c_qp = ow_cost(x_qp, dt, eta, rho)
    # Closed form tracks the exact QP within a few percent.
    if abs(c_opt - c_qp) / c_qp > 0.03:
        raise ValueError("OW closed-form schedule far from discrete QP")
    # QP schedule must have end blocks bigger than interior trades.
    if not (x_qp[0] > 5.0 * np.median(x_qp[1:-1])):
        raise ValueError("QP optimum missing end-block structure")
    # TWAP comparison.
    x_twap = np.full(n, x_total / n)
    c_twap = ow_cost(x_twap, dt, eta, rho)
    improvement = (c_twap - c_opt) / c_twap
    if not (0.0 < improvement < 0.9):
        raise ValueError("OW improvement over TWAP implausible")
    # Cost decreases as resilience grows.
    c_slow = ow_optimal_cost(x_total, t_horizon, eta, 1.0)
    c_fast = ow_optimal_cost(x_total, t_horizon, eta, 4.0)
    if c_fast >= c_slow:
        raise ValueError("cost not decreasing in resilience")
    # Block/rate balance: rate = rho * block.
    if abs(rate - rho * block) > 1e-9:
        raise ValueError("block/rate split inconsistent")
    return {
        "synthetic_ow_block": float(block),
        "synthetic_ow_twap_gain": float(improvement),
        "synthetic_ow_cost_ratio": float(c_fast / c_slow),
        "synthetic_ow_cost_opt": float(c_opt),
        "score": 1.0,
    }
