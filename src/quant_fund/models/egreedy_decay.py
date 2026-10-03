"""Epsilon-greedy exploration schedules: linear, const, 1/t decay.

Theory (Auer/Cesa-Bianchi/Fischer 2002): constant-eps keeps linear
regret; eps_t = min(1, cK/(d^2 t)) with c tuned gets logarithmic regret.
Bench: simulated pseudo-regret for the three schedules — the decay
schedule must beat constant eps, and its regret/log(T) should be small.
"""

import numpy as np

from quant_fund.models._rlt_synth import ARM_MEANS, T_BANDIT


def _run(seed: int, eps_fn) -> float:
    rng = np.random.default_rng(seed)
    k = len(ARM_MEANS)
    pulls = np.zeros(k)
    rewards = np.zeros(k)
    reg = 0.0
    for t in range(1, T_BANDIT + 1):
        eps = eps_fn(t)
        if t <= k or rng.random() < eps:
            a = int(rng.integers(k))
        else:
            a = int(np.argmax(rewards / pulls))
        pulls[a] += 1
        rewards[a] += rng.binomial(1, ARM_MEANS[a])
        reg += ARM_MEANS.max() - ARM_MEANS[a]
    return reg


def bench_egreedy_decay(seed: int = 4605) -> dict[str, float]:
    d2 = (ARM_MEANS.max() - np.sort(ARM_MEANS)[-2]) ** 2
    reg_const = np.mean([_run(seed + i, lambda t: 0.1) for i in range(3)])
    reg_lin = np.mean(
        [_run(seed + 10 + i, lambda t: max(0.01, 0.2 - t / T_BANDIT * 0.2)) for i in range(3)]
    )
    reg_decay = np.mean(
        [_run(seed + 20 + i, lambda t: min(1.0, 5 * len(ARM_MEANS) / (d2 * t))) for i in range(3)]
    )
    return {
        "synthetic_eg_const": float(reg_const),
        "synthetic_eg_linear": float(reg_lin),
        "synthetic_eg_decay": float(reg_decay),
        "synthetic_eg_decay_gain": float(reg_const - reg_decay),
        "synthetic_eg_decay_rate": float(reg_decay / np.log(T_BANDIT)),
    }
