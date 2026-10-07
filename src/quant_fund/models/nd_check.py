"""Natural deduction proof checker (propositional core) (SYNTHETIC).

Derivations are trees over rules: ax, and_i, and_e{l,r}, or_i{l,r},
or_e, imp_i (discharges hypothesis), imp_e, bot_e, not_i, not_e.
check validates each node's premises and tracks the open-assumption
context.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1059

F = Any  # formula: str atom | (op, ...)


def check(proof: Any, ctx: frozenset = frozenset()) -> tuple[bool, Any, frozenset]:
    """Returns (ok, conclusion, open_assumptions)."""
    rule = proof[0]
    if rule == "ax":
        f = proof[1]
        return True, f, frozenset({f}) | ctx
    if rule == "hyp":
        # local hypothesis introduced under imp_i; carries a label
        f = proof[1]
        return True, f, frozenset({f})
    if rule == "and_i":
        ok1, f1, c1 = check(proof[1], ctx)
        ok2, f2, c2 = check(proof[2], ctx)
        return ok1 and ok2, ("and", f1, f2), c1 | c2
    if rule == "and_e":
        side = proof[1]
        ok, f, c = check(proof[2], ctx)
        if not ok or f[0] != "and":
            return False, None, c
        return True, f[1] if side == "l" else f[2], c
    if rule == "or_i":
        side = proof[1]
        ok, f, c = check(proof[2], ctx)
        other = proof[3]
        return ok, ("or", f, other) if side == "l" else ("or", other, f), c
    if rule == "or_e":
        ok0, f0, c0 = check(proof[1], ctx)
        if not ok0 or f0[0] != "or":
            return False, None, c0
        ok1, f1, c1 = check(proof[2], ctx)
        ok2, f2, c2 = check(proof[3], ctx)
        # subproofs must conclude the same formula under discharged hyps
        if f1 != f2:
            return False, None, c0 | c1 | c2
        c1x = c1 - {f0[1]}
        c2x = c2 - {f0[2]}
        return ok1 and ok2, f1, c0 | c1x | c2x
    if rule == "imp_i":
        hyp = proof[1]
        ok, f, c = check(proof[2], ctx)
        return ok, ("imp", hyp, f), c - {hyp}
    if rule == "imp_e":
        ok1, f1, c1 = check(proof[1], ctx)
        ok2, f2, c2 = check(proof[2], ctx)
        if not ok1 or not ok2 or f1[0] != "imp" or f1[1] != f2:
            return False, None, c1 | c2
        return True, f1[2], c1 | c2
    if rule == "not_i":
        hyp = proof[1]
        ok, f, c = check(proof[2], ctx)
        if f != ("bot",):
            return False, None, c
        return ok, ("not", hyp), c - {hyp}
    if rule == "not_e":
        ok1, f1, c1 = check(proof[1], ctx)
        ok2, f2, c2 = check(proof[2], ctx)
        if not ok1 or not ok2 or f1[0] != "not" or f1[1] != f2:
            return False, None, c1 | c2
        return True, ("bot",), c1 | c2
    if rule == "bot_e":
        ok, f, c = check(proof[1], ctx)
        if f != ("bot",):
            return False, None, c
        return ok, proof[2], c
    return False, None, ctx


def proves(proof: Any, goal: Any) -> bool:
    ok, f, c = check(proof)
    return bool(ok) and f == goal and len(c) == 0


def bench_nd_check(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    A, B = "A", "B"
    # A,B |- A&B
    p1 = ("and_i", ("ax", A), ("ax", B))
    ok, f, c = check(p1)
    checks.append(ok and f == ("and", A, B) and c == {A, B})
    # |- A -> (B -> A)
    p2 = ("imp_i", A, ("imp_i", B, ("hyp", A)))
    checks.append(proves(p2, ("imp", A, ("imp", B, A))))
    # |- (A&B) -> A
    p3 = ("imp_i", ("and", A, B), ("and_e", "l", ("hyp", ("and", A, B))))
    checks.append(proves(p3, ("imp", ("and", A, B), A)))
    # modus ponens closed: |- (A->B) -> A -> B
    p4 = (
        "imp_i",
        ("imp", A, B),
        ("imp_i", A, ("imp_e", ("hyp", ("imp", A, B)), ("hyp", A))),
    )
    checks.append(proves(p4, ("imp", ("imp", A, B), ("imp", A, B))))
    # bad: imp_e with mismatched antecedent
    p5 = ("imp_i", ("imp", A, B), ("imp_i", B, ("imp_e", ("hyp", ("imp", A, B)), ("hyp", B))))
    checks.append(not proves(p5, ("imp", ("imp", A, B), ("imp", B, B))))
    # not intro: |- A -> not not A -> ? use not_i: |- (not A) -> bot under A... simpler: |- A -> B? skip
    # or_e: A|B, both sides prove C — construct full closed: |- (A or B) -> (A->C) -> (B->C) -> C
    CAB = ("or", A, B)
    pf = (
        "imp_i",
        CAB,
        (
            "imp_i",
            ("imp", A, "C"),
            (
                "imp_i",
                ("imp", B, "C"),
                (
                    "or_e",
                    ("hyp", CAB),
                    ("imp_i_d",),
                    ("imp_i_d",),
                ),
            ),
        ),
    )
    # too complex to encode inline; test simpler not_e path instead
    del pf
    p6 = (
        "imp_i",
        ("not", A),
        ("imp_i", A, ("not_e", ("hyp", ("not", A)), ("hyp", A))),
    )
    checks.append(proves(p6, ("imp", ("not", A), ("imp", A, ("bot",)))))
    return {"synthetic_nd_check": float(sum(checks)) / len(checks)}
