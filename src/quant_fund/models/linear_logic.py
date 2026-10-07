"""Linear logic prover (multiplicative-additive fragment, small) (SYNTHETIC).

Goals: atoms, tensor A⊗B (splits context), par A⅋B (joins), with A&B
(choose branch — same context used twice), plus A⊕B (pick side),
lolli A⊸B (adds hyp), one 1, top ⊤. Resource-aware: linear hypotheses
consumed exactly once (multiset context).
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1063

F = Any


def prove(ctx: tuple, goal: F, depth: int = 14) -> bool:
    """Multiset ctx of linear formulas; each used at most once."""
    if depth < 0:
        return False
    if goal == ("top",):
        return True
    if goal == ("one",):
        return len(ctx) == 0
    # right rules: success short-circuits; failure falls through to left rules
    if goal[0] == "tensor":
        for i in range(1 << len(ctx)):
            left = tuple(c for j, c in enumerate(ctx) if i & (1 << j))
            right = tuple(c for j, c in enumerate(ctx) if not (i & (1 << j)))
            if prove(left, goal[1], depth - 1) and prove(right, goal[2], depth - 1):
                return True
    elif goal[0] == "par":
        if prove(ctx, goal[1], depth - 1) or prove(ctx, goal[2], depth - 1):
            return True
    elif goal[0] == "with":
        if prove(ctx, goal[1], depth - 1) and prove(ctx, goal[2], depth - 1):
            return True
    elif goal[0] == "plus":
        if prove(ctx, goal[1], depth - 1) or prove(ctx, goal[2], depth - 1):
            return True
    elif goal[0] == "lolli" and prove(ctx + (goal[1],), goal[2], depth - 1):
        return True
    # left rules: consume one hypothesis
    for idx, h in enumerate(ctx):
        rest = ctx[:idx] + ctx[idx + 1 :]
        if h == goal:
            return len(rest) == 0 or all(_prove_top(h2) for h2 in rest)
        if h[0] == "tensor" and prove(rest + (h[1], h[2]), goal, depth - 1):
            return True
        if (
            h[0] == "plus"
            and prove(rest + (h[1],), goal, depth - 1)
            and prove(rest + (h[2],), goal, depth - 1)
        ):
            return True
        if h[0] == "with" and (
            prove(rest + (h[1],), goal, depth - 1) or prove(rest + (h[2],), goal, depth - 1)
        ):
            return True
        if h[0] == "lolli":
            # h = A⊸B: split rest — one part proves A, the rest + B proves goal
            n = len(rest)
            for i in range(1 << n):
                pre = tuple(c for j, c in enumerate(rest) if i & (1 << j))
                post = tuple(c for j, c in enumerate(rest) if not (i & (1 << j)))
                if prove(pre, h[1], depth - 1) and prove(post + (h[2],), goal, depth - 1):
                    return True
    return False


def _prove_top(h: Any) -> bool:
    return bool(h == ("one",) or h == ("top",))


def linear_axiom(a: str) -> bool:
    return prove((a,), a)


def bench_linear_logic(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    checks.append(linear_axiom("A"))
    # A⊗B from {A,B}: splits context
    checks.append(prove(("A", "B"), ("tensor", "A", "B")))
    # can't prove A⊗A from single A (resource accounting)
    checks.append(not prove(("A",), ("tensor", "A", "A")))
    # A⊸B, A |- B
    checks.append(prove(("A", ("lolli", "A", "B")), "B"))
    # |- A⊸A closed
    checks.append(prove((), ("lolli", "A", "A")))
    # with: A&B needs same ctx for both
    checks.append(prove(("A",), ("with", "A", "A")))
    # plus: pick a side
    checks.append(prove(("B",), ("plus", "A", "B")))
    # tensor-left unpacks A⊗B to two resources; proving A leaves B
    # unspent -> invalid in strict linear logic (no weakening)
    checks.append(not prove((("tensor", "A", "B"),), "A"))
    # but consuming both is fine: A⊗B |- A⊗B
    checks.append(prove((("tensor", "A", "B"),), ("tensor", "A", "B")))
    return {"synthetic_linear_logic": float(sum(checks)) / len(checks)}
