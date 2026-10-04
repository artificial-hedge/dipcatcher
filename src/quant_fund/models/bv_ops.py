"""Fixed-width bitvector semantics + tiny solver (SYNTHETIC bench)."""

from __future__ import annotations


def mask(w: int) -> int:
    return (1 << w) - 1


def norm(x: int, w: int) -> int:
    return x & mask(w)


def bv_add(a: int, b: int, w: int) -> int:
    return (a + b) & mask(w)


def bv_mul(a: int, b: int, w: int) -> int:
    return (a * b) & mask(w)


def bv_and(a: int, b: int, w: int) -> int:
    return a & b & mask(w)


def bv_or(a: int, b: int, w: int) -> int:
    return (a | b) & mask(w)


def bv_xor(a: int, b: int, w: int) -> int:
    return (a ^ b) & mask(w)


def bv_not(a: int, w: int) -> int:
    return (~a) & mask(w)


def bv_shl(a: int, k: int, w: int) -> int:
    return (a << k) & mask(w)


def bv_lshr(a: int, k: int, w: int) -> int:
    return (a >> k) & mask(w)


def bv_ult(a: int, b: int, w: int) -> bool:
    return norm(a, w) < norm(b, w)


def bv_slt(a: int, b: int, w: int) -> bool:
    sign = 1 << (w - 1)
    sa = norm(a, w) - (1 << w) if a & sign else norm(a, w)
    sb = norm(b, w) - (1 << w) if b & sign else norm(b, w)
    return sa < sb


def bv_concat(a: int, wa: int, b: int, wb: int) -> int:
    return (norm(a, wa) << wb) | norm(b, wb)


def bv_extract(a: int, hi: int, lo: int) -> int:
    return (a >> lo) & mask(hi - lo + 1)


def solve_enum(clauses: list, widths: list[int]) -> list[int] | None:
    """Brute-force solve clauses over BV vars (widths <= ~8 practical).

    clauses: list of (lhs_expr(vars), op, rhs) where op in {"=","!=","<u"}.
    lhs_expr is a callable over the assignment tuple.
    """
    import itertools

    doms = [range(1 << w) for w in widths]
    for combo in itertools.product(*doms):
        if all(_sat(c, combo) for c in clauses):
            return list(combo)
    return None


def _sat(c: tuple, xs: tuple) -> bool:
    lhs, op, rhs = c
    v = lhs(xs)
    if op == "=":
        return bool(v == rhs)
    if op == "!=":
        return bool(v != rhs)
    if op == "<u":
        return bool(v < rhs)
    raise ValueError(op)


def _bench_bv_ops(seed: int = 0) -> float:
    del seed
    checks = []
    checks.append(bv_add(255, 1, 8) == 0)
    checks.append(bv_mul(16, 16, 8) == 0)
    checks.append(bv_not(0, 4) == 0xF)
    checks.append(bv_slt(0x80, 0x7F, 8))
    checks.append(bv_concat(0xA, 4, 0xB, 4) == 0xAB)
    checks.append(bv_extract(0xB4, 5, 2) == 0xD)
    sol = solve_enum([(lambda xs: bv_add(xs[0], 1, 4), "=", 0), (lambda xs: xs[0], "<u", 16)], [4])
    checks.append(sol == [15])
    sol2 = solve_enum(
        [
            (lambda xs: bv_and(xs[0], 0b1100, 4), "=", 0b0100),
            (lambda xs: xs[0] & 0b0011, "=", 0b0010),
        ],
        [4],
    )
    checks.append(sol2 == [0b0110])
    return sum(checks) / len(checks)


def bench_bv_ops(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bv_ops": _bench_bv_ops(seed)}
