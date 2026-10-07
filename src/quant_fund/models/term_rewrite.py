"""Ordered term rewriting: rules applied leftmost-outermost to normal form, (SYNTHETIC)
plus a local-confluence checker over overlapping redexes.

Terms are nested tuples ("f", a, b) or atoms; variables in patterns are
("?", "name"). Rewriting is terminating by an explicit size bound (workbench
decides confluence by joinability of critical pairs on small systems).
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1003

Term = Any
Subst = dict[str, Term]


def _is_var(t: Term) -> bool:
    return isinstance(t, tuple) and len(t) == 2 and t[0] == "?"


def match(pat: Term, t: Term, s: Subst | None = None) -> Subst | None:
    s = {} if s is None else dict(s)
    if _is_var(pat):
        if pat[1] in s:
            return s if s[pat[1]] == t else None
        s[pat[1]] = t
        return s
    if not isinstance(pat, tuple) or not isinstance(t, tuple):
        return s if pat == t else None
    if len(pat) != len(t) or pat[0] != t[0]:
        return None
    for pa, ta in zip(pat[1:], t[1:], strict=True):
        s = match(pa, ta, s)
        if s is None:
            return None
    return s


def instantiate(pat: Term, s: Subst) -> Term:
    if _is_var(pat):
        return s[pat[1]]
    if isinstance(pat, tuple):
        return tuple(instantiate(a, s) for a in pat)
    return pat


def rewrite_step(t: Term, rules: list[tuple[Term, Term]]) -> Term | None:
    """Leftmost-outermost single rewrite; None if t is in normal form."""
    for lhs, rhs in rules:
        s = match(lhs, t)
        if s is not None:
            return instantiate(rhs, s)
    if isinstance(t, tuple) and len(t) > 1:
        for i, a in enumerate(t[1:], 1):
            r = rewrite_step(a, rules)
            if r is not None:
                return t[:i] + (r,) + t[i + 1 :]
    return None


def normalize(t: Term, rules: list[tuple[Term, Term]], limit: int = 500) -> Term:
    for _ in range(limit):
        r = rewrite_step(t, rules)
        if r is None:
            return t
        t = r
    raise ValueError("non-terminating rewrite")


def _terms_of_depth(
    vars_: list[Term], consts: list[Term], fns: list[str], depth: int
) -> list[Term]:
    if depth == 0:
        return vars_ + consts
    prev = _terms_of_depth(vars_, consts, fns, depth - 1)
    out = list(prev)
    for f in fns:
        for a in prev:
            for b in prev:
                out.append((f, a, b))
    return out


def confluent_on(rules: list[tuple[Term, Term]], samples: list[Term]) -> bool:
    """Check that every sampled term has a unique normal form under all
    first-step redex choices (local confluence on a finite sample)."""
    for t in samples:
        nf = normalize(t, rules)
        for alt in _all_steps_at(t, rules):
            if normalize(alt, rules) != nf:
                return False
    return True


def _all_steps_at(t: Term, rules: list[tuple[Term, Term]]) -> list[Term]:
    out: list[Term] = []
    for lhs, rhs in rules:
        s = match(lhs, t)
        if s is not None:
            out.append(instantiate(rhs, s))
    if isinstance(t, tuple) and len(t) > 1:
        for i, a in enumerate(t[1:], 1):
            for r in _all_steps_at(a, rules):
                out.append(t[:i] + (r,) + t[i + 1 :])
    return out


def bench_term_rewrite(seed: int = _SEED) -> dict[str, float]:
    del seed
    x = ("?", "x")
    rules = [
        (("*", x, ("lit", 1)), x),
        (("*", ("lit", 1), x), x),
        (("inv", ("inv", x)), x),
        (("inv", ("*", x, ("?", "y"))), ("*", ("inv", x), ("inv", ("?", "y")))),
        (("inv", ("lit", 1)), ("lit", 1)),
    ]
    a = ("v", "a")
    b = ("v", "b")
    checks: list[bool] = []
    # ((a^{-1})^{-1} * 1) * (b^{-1})^{-1} -> a * b
    t = ("*", ("*", ("inv", ("inv", a)), ("lit", 1)), ("inv", ("inv", b)))
    checks.append(normalize(t, rules) == ("*", a, b))
    # inv(a*b) rewrites once then stops (vars abstract)
    checks.append(normalize(("inv", ("*", a, b)), rules) == ("*", ("inv", a), ("inv", b)))
    # unit: inv(1*a) -> inv(a)
    checks.append(normalize(("inv", ("*", ("lit", 1), a)), rules) == ("inv", a))
    # confluence on small sample space
    sample = _terms_of_depth([a, b], [("lit", 1)], ["*", "inv"], 1)
    checks.append(confluent_on(rules, sample[:40]))
    # a non-confluent system: a rewrites to two distinct normal forms
    bad: list[tuple[Term, Term]] = [(a, b), (a, ("v", "c"))]
    checks.append(not confluent_on(bad, [a]))
    return {"synthetic_term_rewrite": float(sum(checks)) / len(checks)}
