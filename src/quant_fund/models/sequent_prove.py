"""Intuitionistic sequent calculus prover (LJ-lite) by backward search (SYNTHETIC).

Sequent Gamma |- G. Rules: ax, and_L/and_R, or_L/or_R, imp_L/imp_R
(imp_R only rule producing implication on the right), top_R, bot_L.
Depth-bounded complete search on small goals.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1060

F = Any


def prove(ctx: frozenset, goal: F, depth: int = 12) -> bool:
    if depth < 0:
        return False
    if goal in ctx:
        return True
    if ("bot",) in ctx:
        return True
    if goal == ("top",):
        return True
    # right rules first
    if goal[0] == "and":
        return prove(ctx, goal[1], depth - 1) and prove(ctx, goal[2], depth - 1)
    if goal[0] == "or":
        return prove(ctx, goal[1], depth - 1) or prove(ctx, goal[2], depth - 1)
    if goal[0] == "imp":
        return prove(ctx | {goal[1]}, goal[2], depth - 1)
    if goal[0] == "not":
        return prove(ctx | {goal[1]}, ("bot",), depth - 1)
    # left rules: try each hypothesis destructively
    for h in ctx:
        if h[0] == "and":
            rest = ctx - {h}
            if prove(rest | {h[1], h[2]}, goal, depth - 1):
                return True
        if h[0] == "or":
            rest = ctx - {h}
            if prove(rest | {h[1]}, goal, depth - 1) and prove(rest | {h[2]}, goal, depth - 1):
                return True
        if h[0] == "imp":
            rest = ctx - {h}
            # imp_L: need rest |- antecedent and rest+consequent |- goal
            if prove(rest, h[1], depth - 1) and prove(rest | {h[2]}, goal, depth - 1):
                return True
        if h[0] == "not":
            rest = ctx - {h}
            if prove(rest, h[1], depth - 1):
                return True
    return False


def bench_sequent_prove(seed: int = _SEED) -> dict[str, float]:
    del seed
    A, B, C = "A", "B", "C"
    checks: list[bool] = []
    checks.append(prove(frozenset({"A"}), "A"))
    checks.append(prove(frozenset(), ("imp", A, A)))
    checks.append(prove(frozenset(), ("imp", A, ("imp", B, A))))
    checks.append(prove(frozenset({("and", A, B)}), A))
    checks.append(prove(frozenset({("or", A, B), ("imp", A, C), ("imp", B, C)}), C))
    checks.append(prove(frozenset(), ("imp", ("not", A), ("imp", A, B))))
    # unprovable: |- A or (not A) is intuitionistically invalid
    checks.append(not prove(frozenset(), ("or", A, ("not", A))))
    # transitivity: {A->B, B->C} |- A->C
    checks.append(prove(frozenset({("imp", A, B), ("imp", B, C)}), ("imp", A, C)))
    return {"synthetic_sequent_prove": float(sum(checks)) / len(checks)}
