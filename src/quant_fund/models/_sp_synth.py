"""Shared fixture for wave-189 self-play/game-AI canon (SYNTHETIC).

Two games:

- Tic-tac-toe: full-game perfect-information benchmark with an exact
  minimax oracle. Policies map the 9-cell board → distribution over
  legal moves. Metrics: win/draw rate vs oracle and vs random.
- Kuhn poker: the canonical 3-card 2-player zero-sum game; infosets
  enumerated once so function-approximation methods (deep CFR, NFSP,
  deep fictitious play) can be exploitability-scored on the same
  scale as the tabular CFR reference in `cfr.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# ---------------- tic-tac-toe ----------------

LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def ttt_legal(s: tuple[int, ...]) -> list[int]:
    return [i for i in range(9) if s[i] == 0]


def ttt_winner(s: tuple[int, ...]) -> int | None:
    for a, b, c in LINES:
        if s[a] != 0 and s[a] == s[b] == s[c]:
            return s[a]
    if all(x != 0 for x in s):
        return 0
    return None


@cache
def minimax_val(s: tuple[int, ...], player: int) -> float:
    """Exact game value for `player` (1 or -1) at state `s`."""
    w = ttt_winner(s)
    if w is not None:
        return 0.0 if w == 0 else (1.0 if w == player else -1.0)
    vals = []
    for a in ttt_legal(s):
        ns = list(s)
        ns[a] = player
        vals.append(-minimax_val(tuple(ns), -player))
    return max(vals) if vals else 0.0


def optimal_move(s: tuple[int, ...], player: int) -> int:
    best, best_a = -2.0, -1
    for a in ttt_legal(s):
        ns = list(s)
        ns[a] = player
        v = -minimax_val(tuple(ns), -player)
        if v > best:
            best, best_a = v, a
    return best_a


def encode_ttt(s: tuple[int, ...], player: int) -> FloatArray:
    return np.array([x * player for x in s], dtype=np.float64)


def play_ttt(
    p1: Callable[[FloatArray, list[int]], int], p2: Callable[[FloatArray, list[int]], int]
) -> int:
    """Play a game; returns 1 (p1 win), -1 (p2 win), 0 (draw)."""
    s: tuple[int, ...] = (0,) * 9
    for player, fn in [(1, p1), (-1, p2)] * 5:
        w = ttt_winner(s)
        if w is not None:
            return w
        legal = ttt_legal(s)
        if not legal:
            return 0
        a = fn(encode_ttt(s, player), legal)
        ns = list(s)
        ns[a] = player
        s = tuple(ns)
    return 0


def ttt_eval(
    policy: Callable[[FloatArray, list[int]], int], games: int = 60
) -> tuple[float, float]:
    """(non-loss rate vs oracle, win-or-draw rate vs random)."""
    rng = np.random.default_rng(0)

    def oracle(e: FloatArray, legal: list[int]) -> int:
        s = tuple(int(x) for x in e)
        return optimal_move(s, 1)

    def rnd(e: FloatArray, legal: list[int]) -> int:
        return int(rng.choice(legal))

    s_or = sum(play_ttt(policy, oracle) >= 0 for _ in range(games)) / games
    # vs random as p1 (wins+draws)
    s_rd = sum(play_ttt(policy, rnd) >= 0 for _ in range(games)) / games
    return float(s_or), float(s_rd)


# ---------------- Kuhn poker ----------------

CARDS = [0, 1, 2]  # J, Q, K


def kuhn_util(hist: str, c1: int, c2: int) -> float:
    """Utility for player 0. Hist over {p,b} chars."""
    # terminal eval
    if hist == "pp":
        return 1.0 if c1 > c2 else -1.0
    if hist == "bb":
        return 2.0 if c1 > c2 else -2.0
    if hist == "bp":
        return 1.0
    if hist == "pbp":
        return -1.0
    if hist == "pbb":
        return 2.0
    raise ValueError(hist)


def kuhn_terminal(h: str) -> bool:
    return h in ("pp", "bb", "bp", "pbp", "pbb")


def kuhn_acts(h: str) -> list[str]:
    return ["p", "b"]


def kuhn_infosets() -> list[tuple[int, int, str]]:
    """All (player, card, hist) infosets reachable."""
    out: list[tuple[int, int, str]] = []
    for player in (0, 1):
        for c in CARDS:
            for h in ("", "p", "b", "pb"):
                # legality: hist length parity = player's turn
                if len(h) % 2 == player and not kuhn_terminal(h + "x" if False else h):
                    out.append((player, c, h))
    return out


def encode_kuhn(player: int, card: int, hist: str) -> FloatArray:
    v = np.zeros(3 + 5)
    v[card] = 1.0
    for i, ch in enumerate(hist[:5]):
        v[3 + i] = 1.0 if ch == "b" else -1.0
    return v


def kuhn_exploit(strat: Callable[[int, int, str], FloatArray]) -> float:
    """Nash exploitability of a strategy fn: BR value sum over players."""

    def br_value(hist: str, c1: int, c2: int, brp: int) -> float:
        if kuhn_terminal(hist):
            u = kuhn_util(hist, c1, c2)
            return u if brp == 0 else -u
        player = len(hist) % 2
        card = c1 if player == 0 else c2
        s = strat(player, card, hist)
        if player == brp:
            # best response
            vals = [br_value(hist + a, c1, c2, brp) for a in kuhn_acts(hist)]
            return max(vals)
        return float(
            sum(s[i] * br_value(hist + a, c1, c2, brp) for i, a in enumerate(kuhn_acts(hist)))
        )

    tot = 0.0
    for brp in (0, 1):
        v = 0.0
        for c1 in CARDS:
            for c2 in CARDS:
                if c1 == c2:
                    continue
                v += br_value("", c1, c2, brp) / 6
        tot += v
    return float(tot)


def kuhn_random_strat() -> Callable[[int, int, str], FloatArray]:
    def s(player: int, card: int, hist: str) -> FloatArray:
        return np.array([0.5, 0.5])

    return s
