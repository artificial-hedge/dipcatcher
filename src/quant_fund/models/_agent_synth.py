"""Synthetic agentic fixture: a small tool-graph (SYNTHETIC).

State = int 0..99; tools = deterministic transitions
t_a: s → (s + a) mod 100 for a in {+7, -3, *2 as +s, −17, +11, ÷2 as s//2}
encoded as 6 actions. Task: reach state ≡ 0 mod 10 from start s0.
An agent with state observation can plan; a blind agent cannot.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

N_STATES = 100
N_ACTIONS = 6

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def transition(s: int, a: int) -> int:
    if a == 0:
        return (s + 7) % N_STATES
    if a == 1:
        return (s - 3) % N_STATES
    if a == 2:
        return (2 * s) % N_STATES
    if a == 3:
        return (s - 17) % N_STATES
    if a == 4:
        return (s + 11) % N_STATES
    return s // 2


def is_goal(s: int) -> bool:
    return s % 10 == 0


def bfs_solution(s0: int, max_len: int = 12) -> list[int] | None:
    from collections import deque

    q: deque[tuple[int, list[int]]] = deque([(s0, [])])
    seen = {s0}
    while q:
        s, path = q.popleft()
        if is_goal(s) and path:
            return path
        if len(path) >= max_len:
            continue
        for a in range(N_ACTIONS):
            ns = transition(s, a)
            if ns not in seen:
                seen.add(ns)
                q.append((ns, [*path, a]))
    return None
