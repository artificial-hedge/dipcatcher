"""Norm and trace of GF(p^n)/GF(p) via field_ext arithmetic (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.field_ext import padd, pdivmod, pmod, pmul


def _mul_mod(a: list[int], b: list[int], modulus: list[int], p: int) -> list[int]:
    return pdivmod(pmul(a, b, p), modulus, p)[1]


def _pow_mod(alpha: list[int], e: int, modulus: list[int], p: int) -> list[int]:
    out = [1]
    base = alpha
    while e:
        if e & 1:
            out = _mul_mod(out, base, modulus, p)
        base = _mul_mod(base, base, modulus, p)
        e >>= 1
    return out


def norm_elem(alpha: list[int], modulus: list[int], p: int, n: int) -> list[int]:
    """N(a) = prod of conjugates a^{p^e}, e=0..n-1."""
    out = [1]
    for e in range(n):
        out = _mul_mod(out, _pow_mod(alpha, p**e, modulus, p), modulus, p)
    return pmod(out, p)


def trace_elem(alpha: list[int], modulus: list[int], p: int, n: int) -> list[int]:
    """T(a) = a + a^p + ... + a^{p^{n-1}}."""
    acc = [0]
    for e in range(n):
        acc = padd(acc, _pow_mod(alpha, p**e, modulus, p), p)
    return pmod(acc, p)


def _bench_norm_trace(seed: int = 0) -> float:
    checks = []
    # GF(4)=GF(2)[x]/(x^2+x+1): N(x) = x * x^2 = x^3 = 1; T(x) = x + x^2 = 1
    checks.append(norm_elem([0, 1], [1, 1, 1], 2, 2) == [1])
    checks.append(trace_elem([0, 1], [1, 1, 1], 2, 2) == [1])
    checks.append(norm_elem([1], [1, 1, 1], 2, 2) == [1])
    checks.append(trace_elem([1], [1, 1, 1], 2, 2) == [0])
    # GF(9): modulus x^2+1 over GF(3); N(x) = x * x^3 = x^4 = (-1)^2 = 1
    checks.append(norm_elem([0, 1], [1, 0, 1], 3, 2) == [1])
    # T(x) = x + x^3 = x(1 + x^2) = x(1-1) = 0
    checks.append(trace_elem([0, 1], [1, 0, 1], 3, 2) == [0])
    return float(sum(checks) / len(checks))


def bench_norm_trace(seed: int = 0) -> dict[str, float]:
    return {"synthetic_norm_trace": _bench_norm_trace(seed)}
