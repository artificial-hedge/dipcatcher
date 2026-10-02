"""NegaScout canon: principal-variation search — first child
searched with the full window, remaining children scouted with a
null window and re-searched on failure — versus plain alpha-beta
on the synthetic subtraction-race DAG. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def race_children(s: int, max_take: int = 3) -> list[int]:
    return [s - m for m in range(1, min(max_take, s) + 1)]


def race_game(max_stones: int, max_take: int = 3) -> tuple[dict[int, list[int]], dict[int, int]]:
    children = {s: race_children(s, max_take) for s in range(1, max_stones + 1)}
    return children, {0: -1}


def negascout(
    state: int,
    children: dict[int, list[int]],
    terminal: dict[int, int],
) -> tuple[int, int]:
    """NegaScout / PVS. Returns (value, nodes visited)."""
    nodes = 0

    def go(s: int, alpha: int, beta: int) -> int:
        nonlocal nodes
        nodes += 1
        if s in terminal:
            return terminal[s]
        best = -2
        first = True
        a = alpha
        for c in children[s]:
            if first:
                # principal variation: full window
                score = -go(c, -beta, -a)
                first = False
            else:
                # scout with null window
                score = -go(c, -a - 1, -a)
                if a < score < beta:
                    # scout failed high: re-search full window
                    score = -go(c, -beta, -score)
            best = max(best, score)
            a = max(a, score)
            if a >= beta:
                break
        return best

    return go(state, -2, 2), nodes


def alpha_beta(
    state: int,
    children: dict[int, list[int]],
    terminal: dict[int, int],
) -> tuple[int, int]:
    """Reference alpha-beta for node-count comparison."""
    nodes = 0

    def go(s: int, alpha: int, beta: int) -> int:
        nonlocal nodes
        nodes += 1
        if s in terminal:
            return terminal[s]
        best = -2
        for c in children[s]:
            best = max(best, -go(c, -beta, -alpha))
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    return go(state, -2, 2), nodes


def bench_negascout(seed: int = 20261231) -> dict[str, float]:
    """PVS vs alpha-beta node counts across several pile sizes."""
    out: dict[str, float] = {}
    nodes_ns = 0
    nodes_ab = 0
    correct = 0
    for s0 in (12, 15, 18, 21):
        children, terminal = race_game(s0)
        v_ns, n_ns = negascout(s0, children, terminal)
        v_ab, n_ab = alpha_beta(s0, children, terminal)
        nodes_ns += n_ns
        nodes_ab += n_ab
        correct += int(v_ns == v_ab)
    out["synthetic_positions"] = 4.0
    out["synthetic_values_match"] = float(correct)
    out["synthetic_nodes_ab"] = float(nodes_ab)
    out["synthetic_nodes_ns"] = float(nodes_ns)
    out["synthetic_ns_node_ratio"] = nodes_ns / max(nodes_ab, 1)
    return out
