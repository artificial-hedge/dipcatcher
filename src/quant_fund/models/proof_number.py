"""Proof-number search canon: best-first AND/OR-DAG proving on
the subtraction-race game — attacker (player to move at the root)
tries to prove a forced win. States are (stones, attacker_to_move)
pairs since a pile can be reached at either parity. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

INF = 1 << 30

State = tuple[int, bool]  # (stones, attacker-to-move)


def race_children(s: int, max_take: int = 3) -> list[int]:
    return [s - m for m in range(1, min(max_take, s) + 1)]


def proof_number_search(
    root_stones: int,
    max_take: int = 3,
    max_expansions: int = 10000,
) -> tuple[bool | None, int]:
    """Best-first proof-number search (Allis).

    OR nodes = attacker to move (any child proving wins);
    AND nodes = defender to move (all children must prove).
    Terminal s=0 is a loss for the player to move there.

    Returns (proved, expansions): True if root pn=0 (forced attacker
    win proved), False if root dsn=0 (disproved), None on budget.
    """
    pn: dict[State, int] = {}
    dn: dict[State, int] = {}
    children: dict[State, list[State]] = {}
    expanded: set[State] = set()

    def expand(st: State) -> None:
        s, is_or = st
        ch = [(c, not is_or) for c in race_children(s, max_take)]
        children[st] = ch
        for c in ch:
            if c[0] == 0:
                # terminal at generation: mover lost
                if c[1]:
                    pn.setdefault(c, INF)
                    dn.setdefault(c, 0)
                else:
                    pn.setdefault(c, 0)
                    dn.setdefault(c, INF)
            else:
                pn.setdefault(c, 1)
                dn.setdefault(c, 1)
        if s == 0:
            # player to move at s=0 already lost
            if is_or:
                pn[st], dn[st] = INF, 0
            else:
                pn[st], dn[st] = 0, INF
        elif is_or:
            pn[st], dn[st] = 1, len(ch)
        else:
            pn[st], dn[st] = len(ch), 1
        expanded.add(st)

    def update(st: State) -> None:
        ch = children[st]
        if st[1]:  # OR
            pn[st] = min(pn[c] for c in ch)
            dn[st] = min(sum(dn[c] for c in ch), INF)
        else:
            pn[st] = min(sum(pn[c] for c in ch), INF)
            dn[st] = min(dn[c] for c in ch)

    def most_proving(st: State) -> State:
        while children.get(st):
            update(st)
            ch = children[st]
            if st[1]:
                st = min(ch, key=lambda c: (pn[c], dn[c]))
            else:
                st = min(ch, key=lambda c: (dn[c], pn[c]))
        return st

    root: State = (root_stones, True)
    expand(root)
    expansions = 0
    while pn[root] != 0 and dn[root] != 0:
        if expansions >= max_expansions:
            return None, expansions
        frontier = most_proving(root)
        if frontier in expanded and children.get(frontier):
            # transposition: frontier already expanded — the update
            # pass below handles it; avoid a stuck loop
            pass
        expand(frontier)
        expansions += 1
        # topological order: children always have fewer stones
        for st in sorted(expanded):
            if children.get(st):
                update(st)
    return pn[root] == 0, expansions


def bench_proof_number(seed: int = 20261231) -> dict[str, float]:
    """Prove/disprove race-game roots of known status."""
    out: dict[str, float] = {}
    # s=31: 31 % 4 = 3 → N-position → attacker has a forced win
    proved31, exp31 = proof_number_search(31)
    # s=32: P-position → attacker loses under optimal defense
    proved32, exp32 = proof_number_search(32)
    out["synthetic_proved_31"] = float(proved31 is True)
    out["synthetic_disproved_32"] = float(proved32 is False)
    out["synthetic_expansions_31"] = float(exp31)
    out["synthetic_expansions_32"] = float(exp32)
    out["synthetic_exact_status"] = float(bool(proved31) and proved32 is False)
    return out
