"""UCB1 regret growth vs the Auer/Cesa-Bianchi/Fischer bound (SYNTHETIC).

UCB1 pseudo-regret satisfies E[R_T] <= 8 * sum_i ln(T)/Delta_i + const.
Bench: simulated UCB1 pseudo-regret at several horizons vs the bound
curve — regret must stay below bound, and the realized slope should
track log-growth (regret(T)/log(T) roughly constant in T).
"""

import numpy as np

from quant_fund.models._rlt_synth import ARM_MEANS, T_BANDIT


def _ucb1(seed: int, t_max: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = len(ARM_MEANS)
    pulls = np.zeros(k)
    rewards = np.zeros(k)
    reg = np.zeros(t_max)
    for t in range(1, t_max + 1):
        if t <= k:
            a = t - 1
        else:
            ucb = rewards / pulls + np.sqrt(2 * np.log(t) / pulls)
            a = int(np.argmax(ucb))
        pulls[a] += 1
        rewards[a] += rng.binomial(1, ARM_MEANS[a])
        reg[t - 1] = (
            reg[t - 2] + (ARM_MEANS.max() - ARM_MEANS[a])
            if t > 1
            else (ARM_MEANS.max() - ARM_MEANS[a])
        )
    return reg


def _bound(t: np.ndarray) -> np.ndarray:
    gaps = ARM_MEANS.max() - ARM_MEANS
    gaps = gaps[gaps > 0]
    return np.asarray(8.0 * np.sum(1.0 / gaps) * np.log(t) + 4.0 * len(gaps))


def bench_ucb_bound(seed: int = 4601) -> dict[str, float]:
    reg = _ucb1(seed, T_BANDIT)
    t = np.arange(1, T_BANDIT + 1)
    bnd = _bound(np.maximum(t, 2))
    margin = float(np.min(bnd - reg))
    slope_early = reg[T_BANDIT // 4 - 1] / np.log(T_BANDIT // 4)
    slope_late = reg[-1] / np.log(T_BANDIT)
    return {
        "synthetic_ucb_regret": float(reg[-1]),
        "synthetic_ucb_bound": float(bnd[-1]),
        "synthetic_ucb_margin": margin,
        "synthetic_ucb_slope_drift": abs(float(slope_late) - float(slope_early)),
        "synthetic_ucb_log_slope": float(slope_late),
    }
