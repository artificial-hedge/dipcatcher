"""Shared fixture for wave-191 exploration canon — hard-exploration (SYNTHETIC)
gridworld: reward sits in the far corner behind a wall of distractor
cells; uniform exploration gets stuck near the start. Shared Q-learning
harness with an intrinsic-bonus hook.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

SIDE = 7
GOAL = (0, SIDE - 1)
DISTRACTORS = {(2, 0), (1, 2)}
WALLS = {(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)}


def s2i(s: tuple[int, int]) -> int:
    return s[0] * SIDE + s[1]


def neighbors(s: tuple[int, int]) -> list[tuple[int, tuple[int, int]]]:
    r, c = s
    out: list[tuple[int, tuple[int, int]]] = []
    for a, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        nr, nc = r + dr, c + dc
        if 0 <= nr < SIDE and 0 <= nc < SIDE and (nr, nc) not in WALLS:
            out.append((a, (nr, nc)))
    return out


def reward(s: tuple[int, int]) -> float:
    if s == GOAL:
        return 1.0
    if s in DISTRACTORS:
        return 0.1
    return 0.0


def state_feat(s: tuple[int, int]) -> FloatArray:
    x = np.zeros(6)
    x[0], x[1] = s[0] / SIDE, s[1] / SIDE
    x[2] = 1.0 if s in DISTRACTORS else 0.0
    x[3] = 1.0 if s == GOAL else 0.0
    x[4] = np.sin(np.pi * s[0] * s[1] / SIDE**2)
    x[5] = 1.0
    return x


def q_learn(
    bonus_fn,
    seed: int = 0,
    episodes: int = 300,
    steps: int = 60,
    lr: float = 0.3,
    gamma: float = 0.95,
    eps: float = 0.1,
) -> tuple[FloatArray, float, float]:
    """Tabular Q-learning with intrinsic bonus. bonus_fn(s, s', step, ep,
    rng) -> float. Returns (Q, coverage, success_rate)."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((SIDE * SIDE, 4))
    visited: set[int] = set()
    succ = 0
    for ep in range(episodes):
        s = (3, 0)
        for st in range(steps):
            row = Q[s2i(s)]
            ai = (
                int(rng.integers(4))
                if rng.random() < eps
                else int(rng.choice(np.flatnonzero(row == row.max())))
            )
            legal = dict(neighbors(s))
            if ai not in legal:
                ai = int(rng.choice(list(legal)))
            a = int(ai)
            sp = legal[a]
            r = reward(sp) + bonus_fn(s, sp, st, ep, rng)
            visited.add(s2i(sp))
            if sp == GOAL:
                succ += 1
            nxt = np.max(Q[s2i(sp)]) if sp != GOAL else 0.0
            Q[s2i(s), a] += lr * (r + gamma * nxt - Q[s2i(s), a])
            s = sp
            if sp == GOAL:
                break
        if hasattr(bonus_fn, "end_episode"):
            bonus_fn.end_episode()
    cover = len(visited) / (SIDE * SIDE - len(WALLS))
    return Q, cover, succ / episodes


def null_bonus(s: tuple[int, int], sp: tuple[int, int], st: int, ep: int, rng) -> float:
    return 0.0
