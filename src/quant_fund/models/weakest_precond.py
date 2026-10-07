"""Weakest-precondition calculus + VC generation over a mini while-language (SYNTHETIC).

wp(assign/seq/if) are the standard syntactic rules; loops take an explicit
invariant and emit the three VCs (init, inductiveness, exit). Validity of
generated VCs is discharged by exhaustive evaluation over bounded integer
domains — honest small-model checking, no SMT dependency.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1017

# Formulas: ("lit",int) ("var",x) ("add",a,b) ("sub",a,b) ("le"/"lt"/"ge"/"gt"/"eq"/"ne",a,b)
#           ("and","or","imp","not", ...) ("forall",var,dom_hint,f) not used — enum instead


def _eval(e: Any, s: dict[str, int]) -> Any:
    tag = e[0]
    if tag == "lit":
        return e[1]
    if tag == "var":
        return s[e[1]]
    if tag == "add":
        return _eval(e[1], s) + _eval(e[2], s)
    if tag == "sub":
        return _eval(e[1], s) - _eval(e[2], s)
    if tag == "not":
        return not bool(_eval(e[1], s))
    m = _eval(e[1], s)
    n = _eval(e[2], s)
    return {
        "le": m <= n,
        "lt": m < n,
        "ge": m >= n,
        "gt": m > n,
        "eq": m == n,
        "ne": m != n,
        "and": bool(m) and bool(n),
        "or": bool(m) or bool(n),
        "imp": (not m) or bool(n),
        "xand": bool(m) and bool(n),
    }[tag]


def _subst(e: Any, x: str, v: Any) -> Any:
    if e[0] == "var":
        return v if e[1] == x else e
    if e[0] in ("lit",):
        return e
    return (e[0], *[_subst(a, x, v) for a in e[1:]])


def wp(prog: Any, post: Any) -> Any:
    tag = prog[0]
    if tag == "assign":
        return _subst(post, prog[1], prog[2])
    if tag == "seq":
        return wp(prog[1], wp(prog[2], post))
    if tag == "if":
        return (
            "and",
            ("imp", prog[1], wp(prog[2], post)),
            ("imp", ("not", prog[1]), wp(prog[3], post)),
        )
    if tag == "while":
        inv = prog[3]
        g, body = prog[1], prog[2]
        return (
            "and",
            inv,
            (
                "and",
                ("imp", ("and", inv, g), wp(body, inv)),
                ("imp", ("and", inv, ("not", g)), post),
            ),
        )
    raise ValueError(f"bad prog {prog}")


def holds_on(f: Any, dom: dict[str, range]) -> bool:
    """Exhaustively evaluate closed formula over domain (bounded model check)."""
    import itertools

    ks = list(dom)
    for vals in itertools.product(*(dom[k] for k in ks)):
        if not bool(_eval(f, dict(zip(ks, vals, strict=True)))):
            return False
    return True


def verify(pre: Any, prog: Any, post: Any, dom: dict[str, range]) -> bool:
    vc = ("imp", pre, wp(prog, post))
    return holds_on(vc, dom)


def bench_weakest_precond(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    x, y = ("var", "x"), ("var", "y")
    # swap via tmp: t:=x; x:=y; y:=t  with post {x=y0 ∧ y=x0} expressed via temps
    prog = (
        "seq",
        ("assign", "t", x),
        ("seq", ("assign", "x", y), ("assign", "y", ("var", "t"))),
    )
    post = ("and", ("eq", x, ("var", "y0")), ("eq", y, ("var", "x0")))
    pre = ("and", ("eq", x, ("var", "x0")), ("eq", y, ("var", "y0")))
    checks.append(verify(pre, prog, post, {v: range(0, 4) for v in ("x", "y", "x0", "y0", "t")}))
    # wrong postcondition must fail
    bad = ("eq", x, ("var", "x0"))
    checks.append(not verify(pre, prog, bad, {v: range(0, 4) for v in ("x", "y", "x0", "y0", "t")}))
    # loop: i:=0; while i<n: i:=i+1  with inv i<=n, post i=n (n fixed 3)
    loop = (
        "seq",
        ("assign", "i", ("lit", 0)),
        (
            "while",
            ("lt", ("var", "i"), ("lit", 3)),
            ("assign", "i", ("add", ("var", "i"), ("lit", 1))),
            ("le", ("var", "i"), ("lit", 3)),
        ),
    )
    checks.append(verify(("lit", 1), loop, ("eq", ("var", "i"), ("lit", 3)), {"i": range(0, 6)}))
    # buggy body (i+=2) breaks the eq post but not the le post
    loop2 = (
        "seq",
        ("assign", "i", ("lit", 0)),
        (
            "while",
            ("lt", ("var", "i"), ("lit", 3)),
            ("assign", "i", ("add", ("var", "i"), ("lit", 2))),
            ("le", ("var", "i"), ("lit", 4)),
        ),
    )
    checks.append(verify(("lit", 1), loop2, ("le", ("var", "i"), ("lit", 4)), {"i": range(0, 6)}))
    return {"synthetic_weakest_precond": float(sum(checks)) / len(checks)}
