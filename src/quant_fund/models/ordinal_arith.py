"""Ordinal arithmetic in Cantor normal form (CNF lists of (exp, coeff)) — SYNTHETIC."""

from __future__ import annotations

CNF = list[tuple["CNF", int]]  # ordinal = sum w^e_i * c_i, e_i descending


def nat(n: int) -> CNF:
    return [([], n)] if n > 0 else []


OMEGA: CNF = [(nat(1), 1)]


def omega_pow(e: CNF, c: int = 1) -> CNF:
    return [(e, c)]


def cmp(a: CNF, b: CNF) -> int:
    for (ea, ca), (eb, cb) in zip(a, b, strict=False):
        c = cmp(ea, eb)
        if c:
            return c
        if ca != cb:
            return 1 if ca > cb else -1
    return (len(a) > len(b)) - (len(a) < len(b))


def add(a: CNF, b: CNF) -> CNF:
    if not b:
        return a
    if not a:
        return b
    eb = b[0][0]
    # drop terms of a with exp < eb
    kept = [(e, c) for (e, c) in a if cmp(e, eb) >= 0]
    if kept and cmp(kept[-1][0], eb) == 0:
        e, c = kept.pop()
        return kept + [(e, c + b[0][1])] + b[1:]
    return kept + b


def mul(a: CNF, b: CNF) -> CNF:
    if not a or not b:
        return []
    ea, ca = a[0]
    out: CNF = []
    for e, c in b:
        if not e:  # finite tail: distribute
            out = add(out, [(ea, ca * c)] + a[1:])
        else:
            out = add(out, [(add(ea, e), c)])
    return out


def _bench_ordinal_arith(seed: int = 0) -> float:
    checks = []
    w = OMEGA
    one = nat(1)
    checks.append(add(one, w) == w)  # 1 + w = w
    checks.append(add(w, one) == [(nat(1), 1), ([], 1)])  # w + 1 > w
    checks.append(cmp(add(w, one), w) > 0)
    w2 = omega_pow(nat(2))
    checks.append(mul(w, w) == w2)  # w * w = w^2
    checks.append(mul(nat(3), w) == w)  # 3 * w = w
    checks.append(mul(w, nat(3)) == [(nat(1), 3)])  # w * 3
    # (w+1)*2 = w*2 + 1
    checks.append(cmp(mul(add(w, one), nat(2)), add(mul(w, nat(2)), one)) == 0)
    return float(sum(checks) / len(checks))


def bench_ordinal_arith(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ordinal_arith": _bench_ordinal_arith(seed)}
