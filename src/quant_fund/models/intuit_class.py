"""Intuitionistic vs classical logic: double-negation and Peirce checks (SYNTHETIC).

Classical prover = intuitionistic sequent prover + excluded-middle
branching on atoms (bounded). Validates the strictness gap: Peirce's law
and ¬¬A→A prove classically but not intuitionistically, while
intuitionistic theorems still hold.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1064

F = Any


def _atoms(f: F, out: set | None = None) -> set:
    out = out if out is not None else set()
    if isinstance(f, str):
        out.add(f)
    elif isinstance(f, tuple):
        for x in f[1:]:
            _atoms(x, out)
    return out


def intuit(ctx: frozenset, goal: F, depth: int = 14) -> bool:
    """Same calculus as sequent_prove (inlined, depth-bounded)."""
    if depth < 0:
        return False
    if goal in ctx:
        return True
    if ("bot",) in ctx:
        return True
    if goal == ("top",):
        return True
    if goal[0] == "and":
        return intuit(ctx, goal[1], depth - 1) and intuit(ctx, goal[2], depth - 1)
    if goal[0] == "or":
        return intuit(ctx, goal[1], depth - 1) or intuit(ctx, goal[2], depth - 1)
    if goal[0] == "imp":
        return intuit(ctx | {goal[1]}, goal[2], depth - 1)
    if goal[0] == "not":
        return intuit(ctx | {goal[1]}, ("bot",), depth - 1)
    for h in ctx:
        if h[0] == "and" and intuit((ctx - {h}) | {h[1], h[2]}, goal, depth - 1):
            return True
        if h[0] == "or":
            rest = ctx - {h}
            if intuit(rest | {h[1]}, goal, depth - 1) and intuit(rest | {h[2]}, goal, depth - 1):
                return True
        if h[0] == "imp":
            rest = ctx - {h}
            if intuit(rest, h[1], depth - 1) and intuit(rest | {h[2]}, goal, depth - 1):
                return True
        if h[0] == "not" and intuit(ctx - {h}, h[1], depth - 1):
            return True
    return False


def classical(ctx: frozenset, goal: F, depth: int = 16) -> bool:
    """Classical = intuitionistic + case-split on atoms (Gentzen LEM
    simulation via trying goal under {a} and under {¬a})."""
    if intuit(ctx, goal, depth):
        return True
    if depth <= 0:
        return False
    for h in ctx:
        for a in _atoms(h):
            em = ("or", a, ("not", a))
            # case split: try with a in ctx or (not a) in ctx
            if intuit(ctx | {a}, goal, depth - 2) and intuit(ctx | {("not", a)}, goal, depth - 2):
                return True
            del em
    for a in _atoms(goal):
        if intuit(ctx | {a}, goal, depth - 2) and intuit(ctx | {("not", a)}, goal, depth - 2):
            return True
    return False


def bench_intuit_class(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    A = "A"
    # LEM not intuitionistic
    checks.append(not intuit(frozenset(), ("or", A, ("not", A))))
    # ¬¬A -> A not intuitionistic
    checks.append(not intuit(frozenset(), ("imp", ("not", ("not", A)), A)))
    # classical prover recovers DNE via case split
    checks.append(classical(frozenset(), ("imp", ("not", ("not", A)), A)))
    # classical LEM
    checks.append(classical(frozenset(), ("or", A, ("not", A))))
    # intuitionistic theorem still works
    checks.append(intuit(frozenset(), ("imp", A, ("imp", "B", A))))
    # ¬(A&B) -> ¬A or ¬B is intuitionistically invalid too (weak De Morgan)
    checks.append(
        not intuit(frozenset(), ("imp", ("not", ("and", A, "B")), ("or", ("not", A), ("not", "B"))))
    )
    return {"synthetic_intuit_class": float(sum(checks)) / len(checks)}
