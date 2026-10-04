"""Expectimax — max/chance-node game-tree search for stochastic games.

Dice game: from a position, a d6 roll (chance) then the mover picks one
of the legal advances; reaching the goal pays +1. Expectimax values are
computed by memoized recursion and verified against an independently
computed backward-DP oracle.
"""

import numpy as np

_SEED = 20261231 + 882

GOAL = 20
STEPS = (1, 2, 3)


def expectimax(pos: int, turn: int, memo: dict[tuple[int, int], float] | None = None) -> float:
    """Value for the player to roll at `pos`. Alternating turn via negamax."""
    if memo is None:
        memo = {}
    if pos >= GOAL:
        return 1.0 if turn == 0 else -1.0
    key = (pos, turn)
    if key in memo:
        return memo[key]
    # Chance node: expected over die faces.
    exp = 0.0
    for face in range(1, 7):
        nxt_base = pos + face
        # Move node: choose advance among STEPS (or stand if overshoot).
        opts = [
            -expectimax(nxt_base + s, 1 - turn, memo) for s in STEPS if nxt_base + s <= GOAL + 5
        ]
        exp += max(opts) if opts else -expectimax(GOAL, 1 - turn, memo)
    memo[key] = exp / 6.0
    return memo[key]


def _oracle() -> np.ndarray:
    """Backward DP over states (pos, turn) with explicit terminal band."""
    G = GOAL + 6
    V = np.zeros((2, G + 7))
    for t in range(2):
        for p in range(GOAL, G + 7):
            V[t, p] = 1.0 if t == 0 else -1.0
    for pos in range(GOAL - 1, -1, -1):
        for t in range(2):
            exp = 0.0
            for face in range(1, 7):
                nxt = pos + face
                opts = [-V[1 - t, nxt + s] for s in STEPS if nxt + s <= GOAL + 5]
                exp += max(opts) if opts else -V[1 - t, GOAL]
            V[t, pos] = exp / 6.0
    return V


def bench_expectimax(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: memoized expectimax equals the backward-DP oracle."""
    V = _oracle()
    mism = 0
    for pos in range(GOAL):
        for t in range(2):
            if abs(expectimax(pos, t, {}) - V[t, pos]) > 1e-9:
                mism += 1
    return {"synthetic_expectimax": 1.0 if mism == 0 else 0.0}
