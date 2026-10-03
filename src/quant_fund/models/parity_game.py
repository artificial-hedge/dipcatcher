"""Parity game solver — Zielonka's recursive algorithm on small arenas.

Game: vertices with owner (0=Even,1=Odd), priorities, edges. solve returns
(win_even, win_odd) partition. Zielonka: pick max priority p, attractor
of its owner, recurse on the two subgames — exponential worst-case but
exact on small arenas.
"""

from __future__ import annotations

_SEED = 20261231 + 1042

V = int


def attractor(
    a_set: set[V], player: int, owner: dict[V, int], succ: dict[V, list[V]], pre: dict[V, list[V]]
) -> set[V]:
    """Player's attractor into a_set: vertices they can force entry from."""
    a = set(a_set)
    changed = True
    while changed:
        changed = False
        for v in list(owner):
            if v in a:
                continue
            if owner[v] == player:
                if any(u in a for u in succ.get(v, [])):
                    a.add(v)
                    changed = True
            else:
                ss = succ.get(v, [])
                if ss and all(u in a for u in ss):
                    a.add(v)
                    changed = True
    return a


def solve(
    verts: set[V], owner: dict[V, int], prio: dict[V, int], succ: dict[V, list[V]]
) -> tuple[set[V], set[V]]:
    if not verts:
        return set(), set()
    p = max(prio[v] for v in verts)
    pl = p % 2  # even priorities -> player 0 wins visits
    pre: dict[V, list[V]] = {v: [] for v in verts}
    for v in verts:
        for u in succ.get(v, []):
            if u in pre:
                pre[u].append(v)
    u_set = {v for v in verts if prio[v] == p}
    a = attractor(u_set, pl, owner, succ, pre)
    rest = verts - a

    def subgame(g: set[V]) -> tuple[dict[V, int], dict[V, int], dict[V, list[V]]]:
        return (
            {v: owner[v] for v in g},
            {v: prio[v] for v in g},
            {v: [u for u in succ.get(v, []) if u in g] for v in g},
        )

    so, sp, ss = subgame(rest)
    w0, w1 = solve(rest, so, sp, ss)
    opp = 1 - pl
    w_opp = w1 if pl == 0 else w0
    if not w_opp:
        # pl wins the entire subgame; A already belongs to pl
        return (verts, set()) if pl == 0 else (set(), verts)
    b = attractor(w_opp, opp, owner, succ, pre)
    rest2 = verts - b
    so2, sp2, ss2 = subgame(rest2)
    w0b, w1b = solve(rest2, so2, sp2, ss2)
    if pl == 0:
        return w0b, w1b | b
    return w0b | b, w1b


def bench_parity_game(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # single even-prio vertex owned by 0 -> player 0 wins
    w0, w1 = solve({0}, {0: 0}, {0: 0}, {0: []})
    checks.append(w0 == {0} and w1 == set())
    # v0(prio0,owner0) -> v1(prio1,owner1) -> v0: cycle; highest prio is 1
    # visited infinitely often -> odd (1) wins
    w0, w1 = solve({0, 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: [1], 1: [0]})
    checks.append(w1 == {0, 1})
    # three-node: v2 prio2 even owned by 0 chooses to stay in even cycle
    verts = {0, 1, 2}
    owner = {0: 0, 1: 1, 2: 0}
    prio = {0: 0, 1: 1, 2: 2}
    succ = {0: [1], 1: [0, 2], 2: [2]}
    w0, w1 = solve(verts, owner, prio, succ)
    # v2 self-loops prio 2 -> even wins; v1 (odd) instead forces the v0<->v1
    # cycle whose max-prio 1 favors odd -> v0,v1 are odd-won
    checks.append(w0 == {2} and w1 == {0, 1})
    # attractor: player0 attractor of {2} includes v2 and v0? v0->v1 only; no
    a = attractor({2}, 0, owner, {v: [] for v in verts} | succ, {})
    checks.append(2 in a)
    # even-owned vertex whose all successors are in set -> in attractor
    owner2 = {0: 1, 1: 0}
    succ2 = {0: [1], 1: [0]}
    a2 = attractor({0}, 0, owner2, succ2, {})
    checks.append(a2 == {0, 1})
    return {"synthetic_parity_game": float(sum(checks)) / len(checks)}
