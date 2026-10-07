"""SYNTHETIC SLR(1) parser — FIRST/FOLLOW, LR(0) item sets, ACTION/GOTO
table, shift-reduce evaluation of arithmetic expressions.

Grammar (augmented): E'→E ; E→E+T | T ; T→T*F | F ; F→(E) | num.
"""

from __future__ import annotations

import random

G: list[tuple[str, tuple[str, ...]]] = [
    ("S", ("E",)),
    ("E", ("E", "+", "T")),
    ("E", ("T",)),
    ("T", ("T", "*", "F")),
    ("T", ("F",)),
    ("F", ("(", "E", ")")),
    ("F", ("n",)),
]
NONTERM = {"S", "E", "T", "F"}


def _first() -> dict[str, set[str]]:
    f: dict[str, set[str]] = {nt: set() for nt in NONTERM}
    changed = True
    while changed:
        changed = False
        for lhs, rhs in G:
            if rhs and rhs[0] not in NONTERM:
                if rhs[0] not in f[lhs]:
                    f[lhs].add(rhs[0])
                    changed = True
            elif rhs and rhs[0] in NONTERM and not f[rhs[0]] <= f[lhs]:
                f[lhs] |= f[rhs[0]]
                changed = True
    return f


def _follow(first: dict[str, set[str]]) -> dict[str, set[str]]:
    fo: dict[str, set[str]] = {nt: set() for nt in NONTERM}
    fo["S"].add("$")
    changed = True
    while changed:
        changed = False
        for lhs, rhs in G:
            for i, sym in enumerate(rhs):
                if sym not in NONTERM:
                    continue
                beta = rhs[i + 1 :]
                if beta and beta[0] not in NONTERM:
                    if beta[0] not in fo[sym]:
                        fo[sym].add(beta[0])
                        changed = True
                elif beta:
                    if not first[beta[0]] <= fo[sym]:
                        fo[sym] |= first[beta[0]]
                        changed = True
                else:
                    if not fo[lhs] <= fo[sym]:
                        fo[sym] |= fo[lhs]
                        changed = True
    return fo


def _closure(items: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(items)
    changed = True
    while changed:
        changed = False
        for ri, dot in list(out):
            lhs, rhs = G[ri]
            if dot < len(rhs) and rhs[dot] in NONTERM:
                for j, (l2, _r2) in enumerate(G):
                    if l2 == rhs[dot] and (j, 0) not in out:
                        out.add((j, 0))
                        changed = True
    return out


def _goto(items: set[tuple[int, int]], sym: str) -> set[tuple[int, int]]:
    return _closure(
        {(ri, dot + 1) for ri, dot in items if dot < len(G[ri][1]) and G[ri][1][dot] == sym}
    )


def build_slr() -> tuple[list[dict[str, tuple[str, int]]], list[dict[str, int]]]:
    first = _first()
    follow = _follow(first)
    c0 = _closure({(0, 0)})
    states = [c0]
    idx = {frozenset(c0): 0}
    edges: list[dict[str, int]] = [{}]
    i = 0
    while i < len(states):
        for sym in NONTERM | {"+", "*", "(", ")", "n"}:
            g = _goto(states[i], sym)
            if not g:
                continue
            fs = frozenset(g)
            if fs not in idx:
                idx[fs] = len(states)
                states.append(g)
                edges.append({})
            edges[i][sym] = idx[fs]
        i += 1
    action: list[dict[str, tuple[str, int]]] = [{} for _ in states]
    for i, st in enumerate(states):
        for sym, dst in edges[i].items():
            if sym not in NONTERM:
                action[i][sym] = ("s", dst)
        for ri, dot in st:
            lhs, rhs = G[ri]
            if dot == len(rhs):
                if lhs == "S":
                    action[i]["$"] = ("acc", 0)
                else:
                    for a in follow[lhs]:
                        action[i][a] = ("r", ri)
    return action, edges


def slr_eval(tokens: list[str]) -> float:
    action, _edges = build_slr()
    toks = tokens + ["$"]
    st_stack = [0]
    val_stack: list[float] = []
    sym_stack: list[str] = []
    i = 0
    while True:
        look = toks[i] if toks[i] in ("+", "*", "(", ")", "$") else "n"
        a = action[st_stack[-1]].get(look)
        if a is None:
            raise ValueError("parse error")
        kind, arg = a
        if kind == "s":
            st_stack.append(arg)
            sym_stack.append(toks[i])
            val_stack.append(float(toks[i]) if toks[i] not in "+*()" else 0.0)
            i += 1
        elif kind == "r":
            lhs, rhs = G[arg]
            k = len(rhs)
            if min(len(val_stack), len(sym_stack), len(st_stack)) < k:
                raise ValueError("reduce underflow: stack shorter than production")
            vals = val_stack[-k:] if k else []
            del val_stack[-k:]
            del sym_stack[-k:]
            del st_stack[-k:]
            if k == 1:
                v = vals[0]
            elif k == 3:
                if rhs[1] == "+":
                    v = vals[0] + vals[2]
                elif rhs[1] == "*":
                    v = vals[0] * vals[2]
                else:  # ( E )
                    v = vals[1]
            else:
                raise ValueError(f"unsupported production arity {k}")
            val_stack.append(v)
            sym_stack.append(lhs)
            st_stack.append(_edges_lookup(st_stack, sym_stack))
        else:
            return val_stack[-1] if val_stack else 0.0


_EDGES_CACHE: list[dict[str, int]] = []


def _edges_lookup(st_stack: list[int], sym_stack: list[str]) -> int:
    _a, edges = build_slr()
    return edges[st_stack[-1]][sym_stack[-1]]


def bench_slr_parser(seed: int = 20261231 + 393) -> dict[str, float]:
    rng = random.Random(seed)
    val = rej = 0
    trials = 40
    for _ in range(trials):
        # n + n * n style with parens
        a, b, c = rng.randrange(1, 9), rng.randrange(1, 9), rng.randrange(1, 9)
        got = slr_eval([str(a), "+", str(b), "*", str(c)])
        val += int(got == a + b * c)
        got2 = slr_eval(["(", str(a), "+", str(b), ")", "*", str(c)])
        val += int(got2 == (a + b) * c)
        try:
            slr_eval([str(a), "+", "*", str(b)])
            rej += 0
        except ValueError:
            rej += 1
    tot = trials * 2
    return {
        "synthetic_shift_reduce_value": float(val / tot),
        "synthetic_rejects_bad": float(rej / trials),
        "synthetic_table_built": 1.0,
    }
