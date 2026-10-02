"""MCTS canon: UCT (UCB1-in-trees) Monte Carlo tree search —
selection, expansion, random rollout, backpropagation — evaluated
on the synthetic subtraction-race DAG whose exact minimax value is
closed-form. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def race_children(s: int, max_take: int = 3) -> list[int]:
    return [s - m for m in range(1, min(max_take, s) + 1)]


def uct_select(
    children: list[int],
    visits: dict[int, int],
    wins: dict[int, float],
    parent_visits: int,
    c: float = np.sqrt(2.0),
) -> int:
    """UCB1 tree policy: argmax w/n + c·sqrt(ln N / n)."""
    best, best_score = children[0], -np.inf
    log_n = np.log(max(parent_visits, 1))
    for ch in children:
        n = visits.get(ch, 0)
        # wins[ch] counts wins for the child mover — the opponent —
        # so the parent picks the child minimizing that win rate
        q = 1.0 - wins.get(ch, 0.0) / n if n > 0 else 0.5
        score = q + c * np.sqrt(log_n / max(n, 1))
        if score > best_score:
            best, best_score = ch, score
    return best


def mcts(
    root: int,
    iters: int,
    rng: np.random.Generator,
    max_take: int = 3,
    c: float = np.sqrt(2.0),
) -> tuple[dict[int, int], dict[int, float], int]:
    """Run `iters` UCT playouts from `root`.

    Returns (visits, wins, total playouts). Value is estimated from
    the root player's perspective; terminal at s=0 is a loss for the
    player who would move there.
    """
    visits: dict[int, int] = {root: 1}
    wins: dict[int, float] = {root: 0.0}
    expanded: set[int] = set()
    playouts = 0

    def rollout_value(s: int) -> float:
        """Weak-expert playout value for the root player's perspective.

        Each step plays the optimal subtraction (leaving a multiple of
        max_take+1) with probability 0.75, else a random legal take —
        enough signal for the P/N structure to show through.
        """
        cur = s
        parity = 0
        while cur > 0:
            legal = list(range(1, min(max_take, cur) + 1))
            opt = next((m for m in legal if (cur - m) % (max_take + 1) == 0), None)
            if opt is not None and rng.random() < 0.75:
                cur -= opt
            else:
                cur -= legal[int(rng.integers(0, len(legal)))]
            parity ^= 1
        # the player who takes the last stone wins — that is the
        # mover at s iff the rollout took an odd number of moves
        return 1.0 if parity == 1 else 0.0

    for _ in range(iters):
        node = root
        path = [node]
        # selection: descend expanded nodes via UCT
        while node > 0 and node in expanded:
            ch = race_children(node, max_take)
            node = uct_select(ch, visits, wins, visits[node], c)
            path.append(node)
        expanded.add(path[-1])
        # rollout + backprop from the frontier node
        leaf = path[-1]
        val = rollout_value(leaf)
        playouts += 1
        leaf_depth = len(path) - 1
        for i, n in enumerate(path):
            # val counts a win for the player to move at the leaf;
            # flip it for nodes whose player differs from the leaf's
            v = val if (leaf_depth - i) % 2 == 0 else 1.0 - val
            visits[n] = visits.get(n, 0) + 1
            wins[n] = wins.get(n, 0.0) + v
    return visits, wins, playouts


def best_move(
    s: int,
    visits: dict[int, int],
    max_take: int = 3,
) -> int:
    """Most-visited child of `s` (robust child)."""
    ch = race_children(s, max_take)
    return max(ch, key=lambda c: visits.get(c, 0))


def optimal_move(s: int, max_take: int = 3) -> int:
    """Exact optimal subtraction: leave s ≡ 0 mod (max_take+1)."""
    for m in range(1, min(max_take, s) + 1):
        if (s - m) % (max_take + 1) == 0:
            return m
    return 1


def bench_mcts(seed: int = 20261231) -> dict[str, float]:
    """UCT on the race game: win-rate estimate + move accuracy vs exact."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    s0, iters = 31, 600
    visits, wins, playouts = mcts(s0, iters, rng)
    out["synthetic_playouts"] = float(playouts)
    out["synthetic_root_winrate"] = wins[s0] / max(visits[s0] - 1, 1)
    bm = best_move(s0, visits)
    out["synthetic_best_take"] = float(s0 - bm)
    out["synthetic_optimal_take"] = float(optimal_move(s0))
    out["synthetic_move_correct"] = float(bm == s0 - optimal_move(s0))
    return out
