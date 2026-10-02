"""PUCT canon: AlphaZero-style tree selection
Q(s,a) + c_puct·P(s,a)·sqrt(N(s))/(1+N(s,a)) with a synthetic
policy prior, on the same subtraction-race DAG. Compares prior-
guided search against uniform-prior UCT. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def race_children(s: int, max_take: int = 3) -> list[int]:
    return [s - m for m in range(1, min(max_take, s) + 1)]


def heuristic_prior(s: int, max_take: int = 3) -> FloatArray:
    """Synthetic 'policy head': softmax favoring optimal subtractions.

    A move leaving s ≡ 0 (mod max_take+1) is winning — the prior
    gives it most of the mass, like a trained policy would.
    """
    ch = race_children(s, max_take)
    logits = np.array(
        [2.0 if c % (max_take + 1) == 0 else 0.0 for c in ch],
        dtype=np.float64,
    )
    e = np.exp(logits - logits.max())
    return np.asarray(e / e.sum(), dtype=np.float64)


def puct_select(
    children: list[int],
    prior: FloatArray,
    visits: dict[int, int],
    wins: dict[int, float],
    parent_visits: int,
    c_puct: float = 1.25,
) -> int:
    """AlphaZero PUCT: argmax Q + c·P·sqrt(N_parent)/(1+N_child)."""
    best, best_score = children[0], -np.inf
    sqrt_n = np.sqrt(max(parent_visits, 1))
    for i, ch in enumerate(children):
        n = visits.get(ch, 0)
        # wins[ch] is from the child mover's (opponent's) perspective
        q = 1.0 - wins.get(ch, 0.0) / n if n > 0 else 0.5
        u = c_puct * float(prior[i]) * sqrt_n / (1.0 + n)
        score = q + u
        if score > best_score:
            best, best_score = ch, score
    return best


def puct_search(
    root: int,
    iters: int,
    rng: np.random.Generator,
    max_take: int = 3,
    c_puct: float = 1.25,
    use_prior: bool = True,
) -> tuple[dict[int, int], dict[int, float]]:
    """PUCT tree search with deterministic evaluations.

    Leaf 'value head' is the exact race-game parity (a stand-in for a
    trained value function): +1 if the position is an N-position for
    the player to move, 0 if a P-position.
    """
    visits: dict[int, int] = {root: 1}
    wins: dict[int, float] = {root: 0.0}
    expanded: set[int] = set()

    def value_head(s: int) -> float:
        return 1.0 if s % (max_take + 1) != 0 else 0.0

    for _ in range(iters):
        node = root
        path = [node]
        while node > 0 and node in expanded:
            ch = race_children(node, max_take)
            prior = (
                heuristic_prior(node, max_take) if use_prior else np.full(len(ch), 1.0 / len(ch))
            )
            node = puct_select(ch, prior, visits, wins, visits[node], c_puct)
            path.append(node)
        leaf = path[-1]
        expanded.add(leaf)
        val = value_head(leaf)
        leaf_depth = len(path) - 1
        for i, n in enumerate(path):
            v = val if (leaf_depth - i) % 2 == 0 else 1.0 - val
            visits[n] = visits.get(n, 0) + 1
            wins[n] = wins.get(n, 0.0) + v
    return visits, wins


def best_move(s: int, visits: dict[int, int], max_take: int = 3) -> int:
    ch = race_children(s, max_take)
    return max(ch, key=lambda c: visits.get(c, 0))


def optimal_move(s: int, max_take: int = 3) -> int:
    for m in range(1, min(max_take, s) + 1):
        if (s - m) % (max_take + 1) == 0:
            return m
    return 1


def bench_puct(seed: int = 20261231) -> dict[str, float]:
    """Prior-guided vs uniform-prior PUCT move selection."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    s0 = 31  # N-position; optimal take = 3 (leaves 28, a P-position)
    visits_p, wins_p = puct_search(s0, 200, rng, use_prior=True)
    visits_u, wins_u = puct_search(s0, 200, rng, use_prior=False)
    take_p = s0 - best_move(s0, visits_p)
    take_u = s0 - best_move(s0, visits_u)
    opt = optimal_move(s0)
    out["synthetic_optimal_take"] = float(opt)
    out["synthetic_prior_take"] = float(take_p)
    out["synthetic_uniform_take"] = float(take_u)
    out["synthetic_prior_correct"] = float(take_p == opt)
    out["synthetic_uniform_correct"] = float(take_u == opt)
    out["synthetic_prior_opt_visits"] = float(visits_p.get(s0 - opt, 0))
    out["synthetic_uniform_opt_visits"] = float(visits_u.get(s0 - opt, 0))
    return out
