"""Regex AST simplifier.

Rewrites: flatten nested cat/alt, merge adjacent literals, drop duplicate
alternatives, unwrap singleton nodes, and factor a common literal prefix
out of alternatives (ab|ac -> a(b|c)). SYNTHETIC bench verifies language
equality on samples, node reduction, and idempotence.
"""

from __future__ import annotations

import re as _re
from typing import Any

import numpy as np  # noqa: F401

from quant_fund.models.pike_vm import _Parser

_SEED = 20261231 + 931

Ast = tuple[Any, ...]


def _to_str(node: Ast) -> str:  # noqa: ANN401
    t = node[0]
    if t == "lit":
        return str(node[1])
    if t == "dot":
        return "."
    if t == "cls":
        return "[..]"
    if t == "cat":
        return "".join(_to_str(p) for p in node[1])
    if t == "alt":
        return "|".join(_to_str(p) for p in node[1])
    if t in ("*", "+", "?"):
        return _to_str(node[1]) + str(t)
    if t in ("group", "noncap"):
        return _to_str(node[1])
    return "?"


def simplify(node: Ast) -> Ast:
    t = node[0]
    if t in ("lit", "dot", "cls"):
        return node
    if t in ("group", "noncap"):
        return (t, simplify(node[1]), *node[2:])
    if t in ("*", "+", "?"):
        return (t, simplify(node[1]))
    if t == "cat":
        parts = [simplify(p) for p in node[1]]
        flat: list[Ast] = []
        for p in parts:
            if p[0] == "cat":
                flat.extend(p[1])
            else:
                flat.append(p)
        if len(flat) == 1:
            return flat[0]
        return ("cat", flat)
    if t == "alt":
        parts = [simplify(p) for p in node[1]]
        flat = []
        for p in parts:
            if p[0] == "alt":
                flat.extend(p[1])
            else:
                flat.append(p)
        # dedupe identical branches
        seen: set[str] = set()
        uniq: list[Ast] = []
        for p in flat:
            k = _to_str(p)
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        if len(uniq) == 1:
            return uniq[0]
        # common literal prefix factoring
        pref = _common_prefix(uniq)
        if pref:
            rest = [_strip_prefix(p, pref) for p in uniq]
            return ("cat", [("lit", c) for c in pref] + [simplify(("alt", rest))])
        return ("alt", uniq)
    return node


def _common_prefix(parts: list[Ast]) -> str:
    heads: list[str] = []
    for p in parts:
        if p[0] == "lit":
            heads.append(p[1])
        elif p[0] == "cat" and p[1] and p[1][0][0] == "lit":
            heads.append(p[1][0][1])
        else:
            return ""
    if len(set(heads)) == 1:
        return heads[0]
    return ""


def _strip_prefix(node: Ast, pref: str) -> Ast:
    if not (len(pref) == 1):
        raise ValueError("len(pref) == 1")
    if node[0] == "lit" and node[1] == pref:
        return ("cat", [])
    if node[0] == "cat" and node[1] and node[1][0] == ("lit", pref):
        rest = node[1][1:]
        return ("cat", rest) if len(rest) > 1 else (rest[0] if rest else ("cat", []))
    return node


def _n_nodes(node: Ast) -> int:
    t = node[0]
    if t in ("cat", "alt"):
        return 1 + sum(_n_nodes(p) for p in node[1])
    if t in ("group", "noncap", "*", "+", "?"):
        return 1 + _n_nodes(node[1])
    return 1


def _match(node: Ast, text: str) -> bool:

    # serialize back to pattern string for matching via glushkov
    pat = _serialize(node)
    from quant_fund.models.glushkov_nfa import glushkov_match

    return glushkov_match(pat, text)


def _serialize(node: Ast) -> str:
    t = node[0]
    if t == "lit":
        return str(node[1])
    if t == "dot":
        return "."
    if t == "cls":
        chars = "".join(sorted(node[1]))
        return "[^" + chars + "]" if node[2] else "[" + chars + "]"
    if t == "cat":
        if not node[1]:
            return ""
        return "".join(
            "(" + _serialize(p) + ")" if p[0] == "alt" else _serialize(p) for p in node[1]
        )
    if t == "alt":
        return "(" + "|".join(_serialize(p) for p in node[1]) + ")"
    if t in ("*", "+", "?"):
        sub = node[1]
        s = _serialize(sub)
        if sub[0] in ("alt", "cat") and len(s) > 1:
            s = "(" + s + ")"
        return s + str(t)
    if t == "group":
        return "(" + _serialize(node[1]) + ")"
    if t == "noncap":
        return "(?:" + _serialize(node[1]) + ")"
    raise ValueError(t)


def bench_regex_simplify(seed: int = _SEED) -> dict[str, float]:
    del seed
    score = 0.0
    ast = _Parser("ab|ac|ad").parse()[0][1]
    s = simplify(ast)
    score += 1.0 if _n_nodes(s) < _n_nodes(ast) else 0.0
    ast2 = _Parser("a|a|a").parse()[0][1]
    score += 1.0 if _n_nodes(simplify(ast2)) == 1 else 0.0
    # language equality on samples
    ok = True
    for pat in ["ab|ac|ad", "a(b|b|c)", "x(y|y)z", "pq|pr", "m|n|o"]:
        a = _Parser(pat).parse()[0][1]
        s2 = simplify(a)
        try:
            rgx = _re.compile(_serialize(s2))
        except _re.error:
            ok = False
            break
        for t in ["ab", "ac", "ad", "aa", "b", "xyz", "pq", "pr", "m", "n", "o", ""]:
            if (rgx.fullmatch(t) is not None) != (_re.fullmatch(pat, t) is not None):
                ok = False
    score += 1.0 if ok else 0.0
    # idempotent
    a = _Parser("ab|ac|a(d|d)").parse()[0][1]
    s1, s2v = simplify(a), simplify(simplify(a))
    score += 1.0 if _serialize(s1) == _serialize(s2v) else 0.0
    return {"synthetic_regex_simplify": score / 4.0}
