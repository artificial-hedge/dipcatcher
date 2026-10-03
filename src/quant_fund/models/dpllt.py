"""Lazy DPLL(T): Boolean skeleton + theory-consistency callback (SYNTHETIC bench)."""

from __future__ import annotations

from typing import Any

Clause = frozenset[int]  # signed atom ids: +k means atom true, -k means false
CNF = list[Clause]


def _unit_prop(clauses: CNF, assign: dict[int, bool]) -> tuple[dict[int, bool], bool]:
    """Boolean propagation; returns (assignment, conflict?)."""
    while True:
        changed = False
        for c in clauses:
            vals = []
            unset = []
            for lit in c:
                v = abs(lit)
                want = lit > 0
                if v in assign:
                    vals.append(assign[v] == want)
                else:
                    unset.append(lit)
            if any(vals):
                continue
            if not unset:
                return assign, True
            if len(unset) == 1:
                lit = unset[0]
                v, want = abs(lit), lit > 0
                if v in assign and assign[v] != want:
                    return assign, True
                if v not in assign:
                    assign = {**assign, v: want}
                    changed = True
        if not changed:
            return assign, False


def solve(
    clauses: CNF,
    n_atoms: int,
    theory_ok: Any,
    depth: int = 12,
) -> dict[int, bool] | None:
    """DPLL(T): search Boolean assignment; check theory consistency on partials.

    theory_ok(partial_assign) -> True if theory-consistent so far, False to
    prune (a real solver would return a lemma clause; we just backtrack).
    """
    assign, conflict = _unit_prop(clauses, {})
    if conflict:
        return None
    return _search(clauses, assign, n_atoms, theory_ok, depth)


def _search(
    clauses: CNF, assign: dict[int, bool], n_atoms: int, theory_ok: Any, depth: int
) -> dict[int, bool] | None:
    if depth <= 0:
        return None
    if not theory_ok(assign):
        return None
    if len(assign) == n_atoms:
        return assign
    for v in range(1, n_atoms + 1):
        if v not in assign:
            break
    for val in (True, False):
        a2 = {**assign, v: val}
        a2, conflict = _unit_prop(clauses, a2)
        if conflict:
            continue
        out = _search(clauses, a2, n_atoms, theory_ok, depth - 1)
        if out is not None:
            return out
    return None


def _bench_dpllt(seed: int = 0) -> float:
    del seed
    checks = []
    # atoms: 1 = (x<=2), 2 = (x>=4); theory says they can't both hold
    ok = solve(
        [frozenset({1}), frozenset({2})],
        2,
        lambda a: not (a.get(1) and a.get(2)),
    )
    checks.append(ok is None)
    ok2 = solve(
        [frozenset({1}), frozenset({-1, 2})],
        2,
        lambda a: True,
    )
    checks.append(ok2 is not None and ok2.get(1) is True and ok2.get(2) is True)
    # only boolean model is {1:T,2:T}, which the theory forbids -> UNSAT
    ok3 = solve(
        [frozenset({1, 2}), frozenset({-1, 2}), frozenset({1, -2})],
        2,
        lambda a: not (a.get(1) and a.get(2)),
    )
    checks.append(ok3 is None)
    ok4 = solve([frozenset({1}), frozenset({-1})], 1, lambda a: True)
    checks.append(ok4 is None)
    # theory: atom1 -> atom2 decided values only (partial-safe check)
    ok5 = solve(
        [frozenset({1}), frozenset({-2})],
        2,
        lambda a: not (a.get(1) is True and a.get(2) is False),
    )
    checks.append(ok5 is None)
    ok6 = solve(
        [frozenset({1, 2}), frozenset({-1, -2})],
        2,
        lambda a: not (a.get(1) is not None and a.get(2) is not None and a.get(1) == a.get(2)),
    )
    checks.append(ok6 is not None and ok6[1] != ok6[2])
    return sum(checks) / len(checks)


def bench_dpllt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dpllt": _bench_dpllt(seed)}
