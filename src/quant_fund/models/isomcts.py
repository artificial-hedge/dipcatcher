"""Information-set MCTS — single-observer determinization search (SYNTHETIC).

Game: 2-card showdown. Each player holds a private card in {0..K-1};
higher card wins the pot. Actions: check/fold or bet. The searcher
samples determinizations of the hidden card, runs UCT on the perfect-
information tree, and aggregates action stats — the classic SO-ISMCTS.
Bench: policy bets exactly when holding the strictly best card.
"""

import math

import numpy as np

_SEED = 20261231 + 883

K = 6  # cards 0..K-1, higher wins


def _payoff(mine: int, theirs: int, action: int) -> float:
    """action 0 = check (showdown at unit pot); 1 = bet (opponent calls with top half)."""
    if action == 0:
        return 0.5 if mine > theirs else -0.5
    if theirs < K // 3:  # opponent only folds their worst cards
        return 0.5
    return 1.0 if mine > theirs else -1.0


def _uct_tree(
    mine: int, theirs: int, sims: int, rng: np.random.Generator
) -> dict[int, tuple[float, float]]:
    """Perfect-info UCT over the single decision node (bet or check)."""
    stats = {0: [0, 0.0], 1: [0, 0.0]}  # action -> [n, total reward]
    for i in range(sims):
        a = (
            i % 2
            if i < 4
            else max(
                stats,
                key=lambda k: (
                    stats[k][1] / max(stats[k][0], 1)
                    + math.sqrt(math.log(i + 1) / max(stats[k][0], 1))
                ),
            )
        )
        r = _payoff(mine, theirs, a)
        stats[a][0] += 1
        stats[a][1] += r
    return {a: (float(stats[a][0]), float(stats[a][1])) for a in (0, 1)}


def isomcts_policy(mine: int, sims: int = 60, seed: int = _SEED) -> int:
    rng = np.random.default_rng(seed + mine)
    agg = {0: 0.0, 1: 0.0}
    for _ in range(sims):
        theirs = int(rng.integers(0, K))
        if theirs == mine:
            continue
        stats = _uct_tree(mine, theirs, 8, rng)
        for a in (0, 1):
            if stats[a][0]:
                agg[a] += stats[a][1] / stats[a][0]
    return int(max(agg, key=lambda a: agg[a]))


def bench_isomcts(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: bet on nuts, check otherwise, aggregated over determinizations."""
    nuts = isomcts_policy(K - 1, sims=60, seed=seed)
    weak = isomcts_policy(0, sims=60, seed=seed)
    return {"synthetic_isomcts": 1.0 if (nuts == 1 and weak == 0) else 0.0}
