"""Alpha-beta canon: negamax minimax, alpha-beta pruning, and
transposition-table-cached search on a synthetic game DAG.

Benchmark game: subtraction race — a pile of `s` stones, players
alternate removing 1..`max_take`; taking the last stone wins.
P-positions are s ≡ 0 (mod max_take + 1), so exact values are
checkable; the state graph is a DAG, so a transposition table
genuinely cuts work. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def race_children(s: int, max_take: int = 3) -> list[int]:
    """Successor stone counts for the subtraction race."""
    return [s - m for m in range(1, min(max_take, s) + 1)]


def race_terminal_value(s: int) -> int | None:
    """Terminal value for the player to move: -1 loss at s=0."""
    return -1 if s == 0 else None


def minimax(
    state: int,
    children: dict[int, list[int]],
    terminal: dict[int, int],
) -> tuple[int, int]:
    """Plain negamax. Returns (value for player to move, nodes visited)."""
    nodes = 0

    def go(s: int) -> int:
        nonlocal nodes
        nodes += 1
        if s in terminal:
            return terminal[s]
        return max(-go(c) for c in children[s])

    return go(state), nodes


def alpha_beta(
    state: int,
    children: dict[int, list[int]],
    terminal: dict[int, int],
) -> tuple[int, int]:
    """Negamax with alpha-beta pruning. Returns (value, nodes visited)."""
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


def alpha_beta_tt(
    state: int,
    children: dict[int, list[int]],
    terminal: dict[int, int],
) -> tuple[int, int]:
    """Alpha-beta with an exact-value transposition table."""
    nodes = 0
    tt: dict[int, int] = {}

    def go(s: int, alpha: int, beta: int) -> int:
        nonlocal nodes
        nodes += 1
        if s in terminal:
            return terminal[s]
        if s in tt:
            return tt[s]
        best = -2
        pruned = False
        for c in children[s]:
            best = max(best, -go(c, -beta, -alpha))
            alpha = max(alpha, best)
            if alpha >= beta:
                pruned = True
                break
        if not pruned:
            tt[s] = best
        return best

    val = go(state, -2, 2)
    return val, nodes


def race_game(max_stones: int, max_take: int = 3) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Build the subtraction-race DAG for stones 0..max_stones."""
    children = {s: race_children(s, max_take) for s in range(1, max_stones + 1)}
    terminal = {0: -1}
    return children, terminal


def exact_value(s: int, max_take: int = 3) -> int:
    """Closed-form race-game value: s ≡ 0 (mod max_take+1) loses."""
    return -1 if s % (max_take + 1) == 0 else 1


def bench_alpha_beta(seed: int = 20261231) -> dict[str, float]:
    """Search-effort comparison on the race DAG at a few depths."""
    out: dict[str, float] = {}
    s0 = 21  # N-position (21 % 4 = 1): exact value +1
    children, terminal = race_game(s0)
    v_mm, n_mm = minimax(s0, children, terminal)
    v_ab, n_ab = alpha_beta(s0, children, terminal)
    v_tt, n_tt = alpha_beta_tt(s0, children, terminal)
    exact = exact_value(s0)
    out["synthetic_minimax_value"] = float(v_mm)
    out["synthetic_exact_value"] = float(exact)
    out["synthetic_ab_correct"] = float(v_ab == v_mm == exact)
    out["synthetic_tt_correct"] = float(v_tt == v_mm)
    out["synthetic_nodes_minimax"] = float(n_mm)
    out["synthetic_nodes_ab"] = float(n_ab)
    out["synthetic_nodes_ab_tt"] = float(n_tt)
    out["synthetic_ab_node_ratio"] = n_ab / max(n_mm, 1)
    out["synthetic_tt_node_ratio"] = n_tt / max(n_mm, 1)
    return out
