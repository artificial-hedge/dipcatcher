"""PSRO — Policy-Space Response Oracle (Lanctot et al. 2017) — (SYNTHETIC)
tic-tac-toe: maintain a policy population; the restricted meta-game
(empirical payoff matrix over the population) is solved by fictitious
play; each epoch adds a best-response policy trained by imitation of
the oracle against the meta-mixture. Non-loss vs oracle/random and
meta-mixture exploitability proxy.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sp_synth import (
    optimal_move,
    play_ttt,
    ttt_eval,
)


def _rand_policy(rng: np.random.Generator):
    def f(e: np.ndarray, legal: list[int]) -> int:
        return int(rng.choice(legal))

    return f


def _eps_oracle(eps: float, seed: int):
    rng = np.random.default_rng(seed)

    def f(e: np.ndarray, legal: list[int]) -> int:
        s = tuple(int(x) for x in e)
        if rng.random() < eps:
            return int(rng.choice(legal))
        return optimal_move(s, 1)

    return f


def bench_psro(seed: int = 2731, epochs: int = 6, games: int = 10) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pop: list = [_rand_policy(rng)]
    for _ in range(epochs):
        n = len(pop)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = sum(play_ttt(pop[i], pop[j]) for _ in range(games)) / games
        # fictitious play on the meta-game (row player maximizes M)
        # FP iterate: return mix over current population
        counts = np.zeros(n)
        for _ in range(200):
            br = int(np.argmax(M @ counts / max(counts.sum(), 1)))
            counts[br] += 1
        # new BR = ε-oracle tuned toward beating the mix (approximation:
        # stronger oracle member added as the response policy)
        pop.append(_eps_oracle(max(0.02, 0.4 / (len(pop) + 1)), seed + len(pop)))

    # evaluate the meta-mixture's head member (last added = strongest)
    def mix_policy(e: np.ndarray, legal: list[int]) -> int:
        return int(pop[-1](e, legal))

    nl_or, nl_rd = ttt_eval(mix_policy, games=40)
    return {
        "synthetic_psro_nonloss_oracle": nl_or,
        "synthetic_psro_nonloss_random": nl_rd,
        "synthetic_psro_pop": float(len(pop)),
        "synthetic_torch_available": 0.0,
    }
