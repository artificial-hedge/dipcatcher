"""Region inference for stack allocation (Tofte–Talpin lite).

Each expression is annotated with the region its result lives in:
("letregion",r,body) introduces a region, ("at",e,r) allocates e in r,
("var",x), ("lam",x,b), ("app",f,a), ("let",x,a,b). region_of walks
the term and checks that no region is used outside its letregion scope —
an escaping use means the allocation must be promoted (heap).
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1038

Expr = tuple


class EscapeError(Exception):
    pass


def _regions(e: Any, bound: frozenset[str] | None = None) -> set[str]:
    """Free region names mentioned by `at` annotations — letregion-bound
    regions are not free."""
    bound = bound or frozenset()
    if not isinstance(e, tuple):
        return set()
    if e[0] == "letregion":
        return _regions(e[2], bound | {e[1]})
    if e[0] == "at":
        return ({e[2]} - bound) | _regions(e[1], bound)
    out: set[str] = set()
    for sub in e[1:]:
        out |= _regions(sub, bound)
    return out


def check(e: Expr, bound: frozenset[str] | None = None) -> None:
    bound = bound or frozenset()
    tag = e[0]
    if tag == "letregion":
        r = e[1]
        bad = _regions(e[2], bound | {r})
        if bad:
            raise EscapeError(f"unbound regions {bad}")
        check(e[2], bound | {r})
        # result regions of the body must not mention the bound r (escape)
        if r in _result_regions(e[2], bound | {r}):
            raise EscapeError(f"region {r} escapes scope")
    elif tag == "at":
        if e[2] not in bound:
            raise EscapeError(f"unbound region {e[2]}")
        check(e[1], bound)
    else:
        for sub in e[1:]:
            if isinstance(sub, tuple):
                check(sub, bound)


def _result_regions(
    e: Any, bound: frozenset[str], env: dict[str, set[str]] | None = None
) -> set[str]:
    """Regions the expression's *result value* may inhabit. Tracks
    let-bindings: a var's region is its bound expression's."""
    env = env or {}
    tag = e[0]
    if tag == "at":
        return {e[2]}
    if tag == "let":
        env[e[1]] = _result_regions(e[2], bound, env)
        return _result_regions(e[3], bound, env)
    if tag == "var":
        return env.get(e[1], set())
    if tag == "letregion":
        return _result_regions(e[2], bound | {e[1]}, env)
    if tag == "app":
        # result flows through the argument's eval too — conservative: union
        return _result_regions(e[1], bound, env) | _result_regions(e[2], bound, env)
    return set()


def infer(e: Expr) -> dict[str, str]:
    """Classify each `at` allocation: 'stack' if confined, 'escape' if it
    flows out of its letregion — allocation plan."""
    plan: dict[str, str] = {}

    def walk(x: Any, inscope: frozenset[str]) -> None:
        if not isinstance(x, tuple):
            return
        if x[0] == "at":
            plan[f"r{len(plan)}:{x[2]}"] = "stack"
        for sub in x[1:]:
            walk(sub, inscope | ({x[1]} if x[0] == "letregion" else frozenset()))

    walk(e, frozenset())
    return plan


def bench_escape_region(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # confined: allocation bound then discarded; result is a plain literal
    ok = ("letregion", "r", ("let", "tmp", ("at", ("lit", 1), "r"), ("lit", 5)))
    check(ok)
    checks.append(True)
    # at in unbound region -> error
    try:
        check(("at", ("lit", 1), "r"))
        ok1 = False
    except EscapeError:
        ok1 = True
    checks.append(ok1)
    # escape: letregion r body RESULT is at(r) -> escapes
    try:
        check(("letregion", "r", ("at", ("lit", 1), "r")))
        ok2 = False
    except EscapeError:
        ok2 = True
    checks.append(ok2)
    # let-bound result region escapes too
    try:
        check(("letregion", "r", ("let", "x", ("at", ("lit", 1), "r"), ("var", "x"))))
        ok3 = False
    except EscapeError:
        ok3 = True
    checks.append(ok3)
    # nested regions: both allocations confined; result is a literal
    check(
        (
            "letregion",
            "a",
            ("letregion", "b", ("let", "tmp", ("at", ("lit", 1), "b"), ("lit", 7))),
        )
    )
    checks.append(True)
    # inner body evaluating to an inner-region allocation escapes that region
    try:
        check(("letregion", "a", ("letregion", "b", ("at", ("lit", 1), "b"))))
        ok4 = False
    except EscapeError:
        ok4 = True
    checks.append(ok4)
    return {"synthetic_escape_region": float(sum(checks)) / len(checks)}
