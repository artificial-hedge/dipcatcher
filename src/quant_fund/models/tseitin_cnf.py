"""Tseitin transformation: boolean formula AST -> equisatisfiable CNF.

Each subformula gets a fresh variable; gate clauses encode the subformula's
local semantics. Equisatisfiability is verified against a truth-table oracle
on small formulas — the CNF is linear in the number of gates rather than
exponential like naive CNF conversion.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1004

Form = Any
Clause = tuple[int, ...]


def _is_leaf(f: Form) -> bool:
    return isinstance(f, str)


def tseitin(f: Form) -> tuple[list[Clause], int, int]:
    """Return (clauses, output_var, n_vars). Formula is SAT iff CNF + out=1 SAT."""
    maxv = 0
    for leaf in _vars(f):
        maxv = max(maxv, int(str(leaf)[1:]))
    n = [maxv]
    clauses: list[Clause] = []

    def fresh() -> int:
        n[0] += 1
        return n[0]

    def enc(node: Form) -> int:
        if _is_leaf(node):
            return int(str(node)[1:])
        tag = node[0]
        out = fresh()
        if tag == "not":
            a = enc(node[1])
            clauses.append((-out, -a))  # out -> ~a
            clauses.append((a, out))  # a -> out
        elif tag == "and":
            a, b = enc(node[1]), enc(node[2])
            clauses.append((-out, a))
            clauses.append((-out, b))
            clauses.append((-a, -b, out))
        elif tag == "or":
            a, b = enc(node[1]), enc(node[2])
            clauses.append((out, -a))
            clauses.append((out, -b))
            clauses.append((a, b, -out))
        elif tag == "imp":
            a, b = enc(node[1]), enc(node[2])
            # out <-> (~a | b)
            clauses.append((-out, -a, b))
            clauses.append((a, out))
            clauses.append((-b, out))
        elif tag == "xor":
            a, b = enc(node[1]), enc(node[2])
            # out <-> (a xor b)
            clauses.append((-a, -b, -out))
            clauses.append((a, b, -out))
            clauses.append((a, -b, out))
            clauses.append((-a, b, out))
        else:
            raise ValueError(tag)
        return out

    top = enc(f)
    clauses.append((top,))
    return clauses, top, n[0]


def _eval(f: Form, env: dict[str, bool]) -> bool:
    if _is_leaf(f):
        return env[f]
    tag = f[0]
    if tag == "not":
        return not _eval(f[1], env)
    if tag == "and":
        return _eval(f[1], env) and _eval(f[2], env)
    if tag == "or":
        return _eval(f[1], env) or _eval(f[2], env)
    if tag == "imp":
        return (not _eval(f[1], env)) or _eval(f[2], env)
    if tag == "xor":
        return _eval(f[1], env) != _eval(f[2], env)
    raise ValueError(tag)


def _vars(f: Form) -> list[str]:
    if _is_leaf(f):
        return [f]
    out: list[str] = []
    for a in f[1:]:
        out.extend(_vars(a))
    return sorted(set(out))


def _sat_brute(f: Form) -> bool:
    vs = _vars(f)
    for i in range(2 ** len(vs)):
        env = {v: bool(i & (1 << j)) for j, v in enumerate(vs)}
        if _eval(f, env):
            return True
    return False


def _sat_cnf(clauses: list[Clause], n: int) -> bool:
    def sat(assign: list[bool], i: int) -> bool:
        if i == len(clauses):
            return True
        c = clauses[i]
        ok = False
        for lit in c:
            v = abs(lit)
            if v < len(assign) and assign[v] == (lit > 0):
                ok = True
                break
        return ok and sat(assign, i + 1)

    for i in range(2**n):
        assign = [False] + [bool(i & (1 << (k - 1))) for k in range(1, n + 1)]
        if sat(assign, 0):
            return True
    return False


def bench_tseitin_cnf(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    forms = [
        ("and", "x1", ("or", "x2", ("not", "x3"))),
        ("xor", ("imp", "x1", "x2"), ("and", "x1", "x2")),
        ("and", ("or", "x1", "x2"), ("and", ("not", "x1"), ("not", "x2"))),
        ("imp", ("and", "x1", "x2"), ("or", "x2", "x3")),
    ]
    for f in forms:
        clauses, _top, n = tseitin(f)
        sat_f = _sat_brute(f)
        sat_c = _sat_cnf(clauses, n)
        checks.append(sat_f == sat_c)
        # linear size: clauses <= 4 * nodes + 1
        nodes = len(_vars(f)) + sum(1 for a in _flat(f) if not _is_leaf(a))
        checks.append(len(clauses) <= 4 * nodes + 1)
    return {"synthetic_tseitin_cnf": float(sum(checks)) / len(checks)}


def _flat(f: Form) -> list[Form]:
    if _is_leaf(f):
        return [f]
    out = [f]
    for a in f[1:]:
        out.extend(_flat(a))
    return out
