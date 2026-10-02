"""SYNTHETIC Brzozowski-derivative regex matcher.

Derivative-based matching on a small regex AST
(lit / concat / alt / star / eps) checked against a memoized recursive
matcher oracle on random patterns and strings.
"""

from __future__ import annotations

import random
from functools import cache

# AST: ("lit", c) ("eps",) ("cat", a, b) ("alt", a, b) ("star", a)
RE = tuple


def nullable(r: RE) -> bool:
    if r[0] == "lit" or r[0] == "null":
        return False
    if r[0] == "eps":
        return True
    if r[0] == "cat":
        return nullable(r[1]) and nullable(r[2])
    if r[0] == "alt":
        return nullable(r[1]) or nullable(r[2])
    return True  # star


def derive(r: RE, c: str) -> RE:
    """Brzozowski derivative D_c(r); uses ("null",) for ∅."""
    if r[0] == "lit":
        return ("eps",) if r[1] == c else ("null",)
    if r[0] == "eps" or r[0] == "null":
        return ("null",)
    if r[0] == "cat":
        d = ("cat", derive(r[1], c), r[2])
        return ("alt", d, derive(r[2], c)) if nullable(r[1]) else d
    if r[0] == "alt":
        return ("alt", derive(r[1], c), derive(r[2], c))
    return ("cat", derive(r[1], c), r)  # star


def matches(r: RE, s: str) -> bool:
    for c in s:
        r = derive(r, c)
    return nullable(r)


def oracle_match(r: RE, s: str) -> bool:
    """Independent matcher: full-string language membership by DP."""

    @cache
    def mm(re_: RE, i: int, j: int) -> bool:
        # does re_ match s[i:j] exactly?
        if re_[0] == "lit":
            return j - i == 1 and s[i] == re_[1]
        if re_[0] == "eps":
            return j == i
        if re_[0] == "null":
            return False
        if re_[0] == "alt":
            return mm(re_[1], i, j) or mm(re_[2], i, j)
        if re_[0] == "cat":
            return any(mm(re_[1], i, k) and mm(re_[2], k, j) for k in range(i, j + 1))
        # star: Kleene closure — partition into ≥0 nonempty pieces matching inner
        if j == i:
            return True
        return any(
            mm(re_[1], i, k) and mm(re_, k, j) for k in range(i + 1, j + 1) if mm(re_[1], i, k)
        )

    return mm(r, 0, len(s))


def _rand_re(rng: random.Random, depth: int) -> RE:
    if depth == 0 or rng.random() < 0.3:
        return rng.choice([("lit", "a"), ("lit", "b"), ("eps",)])
    op = rng.choice(["cat", "alt", "star"])
    if op == "star":
        return ("star", _rand_re(rng, depth - 1))
    return (op, _rand_re(rng, depth - 1), _rand_re(rng, depth - 1))


def bench_brzozowski_deriv(seed: int = 20261231 + 502) -> dict[str, float]:
    rng = random.Random(seed)
    agree = 0
    n = 120
    for _ in range(n):
        r = _rand_re(rng, 3)
        s = "".join(rng.choice("ab") for _ in range(rng.randrange(0, 7)))
        agree += int(matches(r, s) == oracle_match(r, s))
    nul = all(nullable(r) == oracle_match(r, "") for r in [_rand_re(rng, 3) for _ in range(60)])
    return {
        "synthetic_match_agrees_oracle": agree / n,
        "synthetic_nullable_correct": float(nul),
    }
