"""Interval abstract interpretation over a mini while-language.

Programs are tuples: ("assign", x, expr), ("seq", s1, s2),
("if", guard_expr, then, else), ("while", guard_expr, body, inv_hint).
Intervals [lo,hi] with None = unbounded; arithmetic lifts to intervals with
natural bound propagation; while-loops iterate the body transformer with
widen(⊥ threshold) then narrow until post-fixpoint. Soundness verified
against concrete traces.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1005

Ivl = tuple[float, float]
State = dict[str, Ivl]

INF = float("inf")


def _iadd(a: Ivl, b: Ivl) -> Ivl:
    return (a[0] + b[0], a[1] + b[1])


def _isub(a: Ivl, b: Ivl) -> Ivl:
    return (a[0] - b[1], a[1] - b[0])


def _imul(a: Ivl, b: Ivl) -> Ivl:
    cands = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(cands), max(cands))


def _iunion(a: Ivl, b: Ivl) -> Ivl:
    return (min(a[0], b[0]), max(a[1], b[1]))


def _eval_expr(e: Any, s: State) -> Ivl:
    if isinstance(e, tuple):
        op = e[0]
        if op == "lit":
            v = float(e[1])
            return (v, v)
        if op == "var":
            return s[e[1]]
        a = _eval_expr(e[1], s)
        b = _eval_expr(e[2], s)
        if op == "add":
            return _iadd(a, b)
        if op == "sub":
            return _isub(a, b)
        if op == "mul":
            return _imul(a, b)
    raise ValueError(e)


def _guard_range(e: Any, s: State, want: bool) -> Ivl:
    """Interval for the truth value of guard (1 true / 0 false) — crude."""
    op = e[0] if isinstance(e, tuple) else None
    if op in ("le", "lt", "ge", "gt", "eq", "ne"):
        return (0.0, 1.0)
    raise ValueError(e)


def _meet_guard(s: State, e: Any, truth: bool) -> State:
    """Refine state under e == truth for simple guards: x <= lit, lit <= x."""
    op, a, b = e
    out = dict(s)
    if isinstance(a, tuple) and a[0] == "var" and isinstance(b, tuple) and b[0] == "lit":
        x = a[1]
        lo, hi = s.get(x, (-INF, INF))
        if truth:
            if op == "le":
                hi = min(hi, float(b[1]))
            elif op == "lt":
                hi = min(hi, float(b[1]) - 1.0)
            elif op == "ge":
                lo = max(lo, float(b[1]))
            elif op == "gt":
                lo = max(lo, float(b[1]) + 1.0)
            elif op == "eq":
                lo = max(lo, float(b[1]))
                hi = min(hi, float(b[1]))
        else:
            if op == "le":
                lo = max(lo, float(b[1]) + 1.0)
            elif op == "lt":
                lo = max(lo, float(b[1]))
            elif op == "ge":
                hi = min(hi, float(b[1]) - 1.0)
            elif op == "gt":
                hi = min(hi, float(b[1]))
            elif op == "eq":
                pass  # x != c leaves interval unchanged on a contiguous domain
        out[x] = (lo, hi)
    return out


def _widen(a: Ivl, b: Ivl) -> Ivl:
    lo = a[0] if b[0] >= a[0] else -INF
    hi = a[1] if b[1] <= a[1] else INF
    return (lo, hi)


def _narrow(a: Ivl, b: Ivl) -> Ivl:
    lo = b[0] if a[0] == -INF else a[0]
    hi = b[1] if a[1] == INF else a[1]
    return (lo, hi)


def transfer(prog: Any, s: State, widen_after: int = 4) -> State:
    op = prog[0]  # type: ignore[index]
    if op == "assign":
        out = dict(s)
        out[prog[1]] = _eval_expr(prog[2], s)  # type: ignore[index]
        return out
    if op == "seq":
        return transfer(prog[2], transfer(prog[1], s), widen_after)  # type: ignore[index]
    if op == "if":
        then_s = transfer(prog[2], _meet_guard(s, prog[1], True), widen_after)  # type: ignore[index]
        else_s = transfer(prog[3], _meet_guard(s, prog[1], False), widen_after)  # type: ignore[index]
        keys = set(then_s) | set(else_s)
        return {k: _iunion(then_s.get(k, (-INF, INF)), else_s.get(k, (-INF, INF))) for k in keys}
    if op == "while":
        guard, body = prog[1], prog[2]  # type: ignore[index]
        cur = _meet_guard(s, guard, True)
        last_joined = cur
        it = 0
        widened = False
        while True:
            post = transfer(body, cur, widen_after)
            # loop-head state: union of previous iterate and body result,
            # re-refined by the guard (keeps x <= bound during iteration)
            joined = {
                k: _iunion(cur.get(k, (-INF, INF)), post.get(k, (-INF, INF)))
                for k in set(cur) | set(post)
            }
            last_joined = joined
            nxt = _meet_guard(joined, guard, True)
            it += 1
            if nxt == cur:
                break
            if it > widen_after and not widened:
                cur = {k: _widen(cur.get(k, (-INF, INF)), nxt.get(k, (-INF, INF))) for k in nxt}
                widened = True
            elif widened:
                cur = {k: _narrow(cur.get(k, (-INF, INF)), nxt.get(k, (-INF, INF))) for k in nxt}
            else:
                cur = nxt
            if it > 40:
                break
        # exit applies guard-false to the pre-guard joined state (x may have
        # just crossed the bound inside the body)
        return _meet_guard(last_joined, guard, False)
    raise ValueError(prog)


def run_concrete(prog: Any, s: dict[str, float], steps: int = 1000) -> dict[str, float]:
    op = prog[0]  # type: ignore[index]
    if op == "assign":
        out = dict(s)
        out[prog[1]] = _ceval(prog[2], s)  # type: ignore[index]
        return out
    if op == "seq":
        return run_concrete(prog[2], run_concrete(prog[1], s, steps), steps)  # type: ignore[index]
    if op == "if":
        return run_concrete(prog[2] if _ceval_guard(prog[1], s) else prog[3], s, steps)  # type: ignore[index]
    if op == "while":
        n = 0
        while _ceval_guard(prog[1], s):  # type: ignore[index]
            s = run_concrete(prog[2], s, steps)  # type: ignore[index]
            n += 1
            if n > steps:
                raise ValueError("nontermination")
        return s
    raise ValueError(prog)


def _ceval(e: Any, s: dict[str, float]) -> float:
    if isinstance(e, tuple):
        if e[0] == "lit":
            return float(e[1])
        if e[0] == "var":
            return s[e[1]]
        a, b = _ceval(e[1], s), _ceval(e[2], s)
        return {"add": a + b, "sub": a - b, "mul": a * b}[e[0]]
    raise ValueError(e)


def _ceval_guard(e: Any, s: dict[str, float]) -> bool:
    op, a, b = e
    av, bv = _ceval(a, s), _ceval(b, s)
    return {
        "le": av <= bv,
        "lt": av < bv,
        "ge": av >= bv,
        "gt": av > bv,
        "eq": av == bv,
        "ne": av != bv,
    }[op]


def bench_interval_analysis(seed: int = _SEED) -> dict[str, float]:
    del seed
    x = ("var", "x")
    one = ("lit", 1.0)
    prog = (
        "seq",
        ("assign", "x", ("lit", 0.0)),
        ("while", ("lt", x, ("lit", 10.0)), ("assign", "x", ("add", x, one))),
    )
    s = transfer(prog, {"x": (-INF, INF)})
    checks: list[bool] = []
    # loop exit: x in [10,10]
    checks.append(s["x"] == (10.0, 10.0))
    # if/else join
    prog2 = (
        "if",
        ("le", x, ("lit", 5.0)),
        ("assign", "y", ("lit", 1.0)),
        ("assign", "y", ("lit", 2.0)),
    )
    s2 = transfer(prog2, {"x": (-INF, INF)})
    checks.append(s2["y"] == (1.0, 2.0))
    # sound vs concrete: x==10 after loop
    s3 = run_concrete(prog, {"x": 0.0})
    checks.append(abs(s3["x"] - 10.0) < 1e-9)
    # interval arithmetic: [-2,3]*[-1,4] = [-8,12]
    checks.append(_imul((-2.0, 3.0), (-1.0, 4.0)) == (-8.0, 12.0))
    # descending loop
    prog4 = (
        "seq",
        ("assign", "x", ("lit", 20.0)),
        ("while", ("gt", x, ("lit", 4.0)), ("assign", "x", ("sub", x, one))),
    )
    s4 = transfer(prog4, {"x": (-INF, INF)})
    checks.append(s4["x"][0] >= 4.0 and s4["x"][1] <= 5.0)
    return {"synthetic_interval_analysis": float(sum(checks)) / len(checks)}
