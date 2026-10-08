"""Well-structured transition systems: coverability via backward search (SYNTHETIC).

States are multisets of places (Petri-net style, ω-values after
acceleration). pre(U) computed upward-closed; acceleration extrapolates
strictly-growing coordinates to ω. Terminates via wqo — the classic
Abdulla/Finkel-Schnoebelen backward algorithm, on bounded nets here.
"""

from __future__ import annotations

_SEED = 20261231 + 1046

Marking = tuple[int | float, ...]
OMEGA = float("inf")


def _leq(a: Marking, b: Marking) -> bool:
    return all(x <= y for x, y in zip(a, b, strict=True))


def _upward(m: Marking, bound: int = 6) -> list[Marking]:
    return [m]


def pre(trans: list[tuple[Marking, Marking]], u_set: list[Marking]) -> list[Marking]:
    """Predecessors of upward-closed U: minimal markings that can step up
    into U. For net transitions (consume,produce): m' = u - produce + consume."""
    out: list[Marking] = []
    for cons, prod in trans:
        for u in u_set:
            m = tuple(
                max(0, int(u[i] if u[i] != OMEGA else 0) - (prod[i] if prod[i] != OMEGA else 0))
                + cons[i]
                for i in range(len(cons))
            )
            out.append(m)
    return out


def _accelerate(m: Marking, older: list[Marking]) -> Marking:
    """If m strictly larger than some older marking in the subsumed set,
    pump the growing coords to ω."""
    for o in older:
        if o != m and _leq(o, m):
            return tuple(OMEGA if m[i] > o[i] else m[i] for i in range(len(m)))
    return m


def coverable(
    trans: list[tuple[Marking, Marking]],
    init: Marking,
    target: Marking,
    bound: int = 400,
) -> bool:
    """Backward coverability: True iff `init` can reach a marking >= target.
    Explores a minimal basis of predecessors (wqo-subsumption + ω-accel);
    answer = some basis element <= init."""
    if _leq(target, init):
        return True
    basis: list[Marking] = []
    work: list[Marking] = [target]
    steps = 0
    while work and steps < bound:
        steps += 1
        m = work.pop()
        m = _accelerate(m, basis)
        if any(_leq(o, m) for o in basis):
            continue
        if _leq(m, init):
            return True
        basis = [o for o in basis if not _leq(m, o)]
        basis.append(m)
        for p_ in pre(trans, [m]):
            if not any(_leq(o, p_) for o in basis):
                work.append(p_)
    return any(_leq(b, init) for b in basis)


def bench_wsts_cover(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # net: t1 consumes (1,0)->produces (0,1); t2 consumes (0,1)->produces (1,0)
    t1 = ((1, 0), (0, 1))
    t2 = ((0, 1), (1, 0))
    # from init (1,0): cover (0,1) via t1 -> True
    checks.append(coverable([t1, t2], (1, 0), (0, 1)))
    # cover (0,2): can't stack two tokens through the 1-token cycle
    checks.append(not coverable([t1, t2], (1, 0), (0, 2)))
    # spawner t3 makes arbitrarily many tokens: (0,3) coverable from empty
    t3 = ((0, 0), (0, 1))
    checks.append(coverable([t3], (0, 0), (0, 3)))
    # acceleration check: older (0,1) vs new (0,3) -> pump to ω
    m = _accelerate((0, 3), [(0, 1)])
    checks.append(m == (0, OMEGA))
    # _leq with ω
    checks.append(_leq((0, 3), (0, OMEGA)))
    return {"synthetic_wsts_cover": float(sum(checks)) / len(checks)}
