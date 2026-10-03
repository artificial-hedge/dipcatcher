"""Repairable-system RAM model: 2-unit parallel + one repair crew.

CTMC on states 2/1/0 up-units (state 0 is absorbing for availability):
Q_{k,k-1} = k*lam (failure), Q_{k,k+1} = mu (repair, single crew).
Steady-state availability A = pi_2 + pi_1; compared against a
discrete-event simulation of the same system.
"""

import numpy as np

from quant_fund.models._rel_synth import RAM_LAM, RAM_MU


def _steady_state(lam: float, mu: float) -> np.ndarray:
    # generator on states k = number of up units, k in {0,1,2}
    q = np.zeros((3, 3))
    for k in (1, 2):
        q[k, k - 1] = k * lam  # failure: k -> k-1
        q[k, k] -= k * lam
    for k in (0, 1):
        q[k, k + 1] = mu  # repair (single crew): k -> k+1
        q[k, k] -= mu
    # pi Q = 0, sum pi = 1 -> replace one row of Q^T with ones
    a = q.T.copy()
    a[0] = 1.0
    b = np.zeros(3)
    b[0] = 1.0
    return np.linalg.solve(a, b)


def _simulate(lam: float, mu: float, horizon: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    state, t, up_time = 2, 0.0, 0.0
    while t < horizon:
        rate = state * lam + (mu if state < 2 else 0.0)
        dt = rng.exponential(1.0 / rate)
        up_time += min(dt, horizon - t) * (1 if state > 0 else 0)
        t += dt
        if t >= horizon:
            break
        p_fail = state * lam / rate
        if rng.uniform() < p_fail:
            state -= 1
        else:
            state += 1
    return up_time / horizon


def bench_ram_markov(seed: int = 4905) -> dict[str, float]:
    pi = _steady_state(RAM_LAM, RAM_MU)
    avail = float(pi[1] + pi[2])  # system up iff >=1 unit up
    mc = _simulate(RAM_LAM, RAM_MU, horizon=30000.0, seed=seed)
    down = float(pi[0])
    return {
        "synthetic_ram_avail": avail,
        "synthetic_ram_avail_mc_err": abs(avail - mc),
        "synthetic_ram_down": down,
        "synthetic_ram_pi2": float(pi[2]),
        "synthetic_ram_mttf": float(pi[1] + pi[2]) / (pi[1] * RAM_LAM + 1e-18),
    }
