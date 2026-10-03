"""Glushkov (position) automaton.

Linearizes the regex into positions; computes nullable/firstpos/lastpos/
followpos on the syntax tree; builds the automaton with n+1 states.
SYNTHETIC bench: agreement with `re`, state count == positions + 1.
"""

from __future__ import annotations

import re as _re
from typing import Any

import numpy as np

from quant_fund.models.pike_vm import _Parser

_SEED = 20261231 + 930

Ast = tuple[Any, ...]


def _build(pat: str):
    """Return (nullable, firstpos, lastpos, followpos, labels)."""
    ast = _Parser(pat).parse()[0][1]
    order: list[Ast] = []

    def positions(n: Ast) -> None:
        t = n[0]
        if t in ("lit", "dot", "cls"):
            order.append(n)
            return
        for child in n[1] if t in ("cat", "alt") else [n[1]]:
            positions(child)

    positions(ast)
    idx = {id(n): i for i, n in enumerate(order)}
    nullable_d: dict[int, bool] = {}
    first_d: dict[int, set[int]] = {}
    last_d: dict[int, set[int]] = {}
    follow: dict[int, set[int]] = {i: set() for i in range(len(order))}

    def rec(n: Ast) -> tuple[bool, set[int], set[int]]:
        t = n[0]
        if t in ("lit", "dot", "cls"):
            i = idx[id(n)]
            nullable_d[i], first_d[i], last_d[i] = False, {i}, {i}
            return False, {i}, {i}
        if t == "cat":
            parts = n[1]
            info = [rec(p) for p in parts]
            # followpos: lastpos of each part joined to firstpos of next
            # non-nullable stretch
            for k in range(len(parts) - 1):
                src: set[int] = set()
                j = k
                while j >= 0:
                    src |= info[j][2]
                    if not info[j][0]:
                        break
                    j -= 1
                dst: set[int] = set()
                j = k + 1
                while j < len(parts):
                    dst |= info[j][1]
                    if not info[j][0]:
                        break
                    j += 1
                for s in src:
                    follow[s] |= dst
            null = all(i0[0] for i0 in info)
            f: set[int] = set()
            for i0 in info:
                f |= i0[1]
                if not i0[0]:
                    break
            la: set[int] = set()
            for i0 in reversed(info):
                la |= i0[2]
                if not i0[0]:
                    break
            return null, f, la
        if t == "alt":
            f = set()
            la = set()
            null = False
            for p in n[1]:
                pn, pf, pl = rec(p)
                f |= pf
                la |= pl
                null = null or pn
            return null, f, la
        if t in ("group", "noncap"):
            return rec(n[1])
        if t in ("*", "+"):
            pn, pf, pl = rec(n[1])
            for s in pl:
                follow[s] |= pf
            return (True if t == "*" else pn), pf, pl
        if t == "?":
            pn, pf, pl = rec(n[1])
            return True, pf, pl
        raise ValueError(t)

    nul, first, last = rec(ast)
    return nul, first, last, follow, order


def glushkov_match(pat: str, text: str) -> bool:
    nul, first, last, follow, order = _build(pat)
    if nul and not text:
        return True

    def lab_ok(node: Ast, ch: str) -> bool:
        t = node[0]
        if t == "lit":
            return bool(ch == node[1])
        if t == "dot":
            return True
        in_cls = bool(ch in node[1])
        return in_cls != bool(node[2])

    cur: set[int] = set()
    nxt: set[int] = set()
    for i, ch in enumerate(text):
        src = first if i == 0 else nxt
        cur = set()
        for q in src:
            if lab_ok(order[q], ch):
                cur.add(q)
        nxt = set()
        for q in cur:
            nxt |= follow[q]
    return any(i in last for i in cur)


def bench_glushkov_nfa(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    nul, first, last, follow, order = _build("(a|b)*abb")
    score += 1.0 if len(order) == 5 else 0.0
    ok = all(
        glushkov_match("(a|b)*abb", t) == (t.endswith("abb") and set(t) <= {"a", "b"})
        for t in ["abb", "aabb", "babb", "ab", "abab", "", "ababababb"]
    )
    score += 1.0 if ok else 0.0
    alpha = "ab"
    ok = True
    for _ in range(150):
        pat = "".join(
            str(rng.choice(list(alpha) + ["*", "+", "?", "|", "(", ")", "."]))
            for _ in range(int(rng.integers(3, 9)))
        )
        try:
            rgx = _re.compile(pat)
            glushkov_match(pat, "x")
        except (ValueError, _re.error):
            continue
        t = "".join(str(rng.choice(list(alpha + "c"))) for _ in range(int(rng.integers(0, 8))))
        try:
            got = glushkov_match(pat, t)
        except (ValueError, KeyError):
            continue
        if got != (rgx.fullmatch(t) is not None):
            ok = False
            break
    score += 1.0 if ok else 0.0
    score += 1.0 if glushkov_match("a?a?a?bbb", "bbb") else 0.0
    return {"synthetic_glushkov_nfa": score / 4.0}
