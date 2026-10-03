"""Separability: f has distinct roots iff gcd(f, f') = 1 (SYNTHETIC)."""

from __future__ import annotations


def _poly_deriv(f: list[float]) -> list[float]:
    return [i * c for i, c in enumerate(f)][1:]


def _poly_gcd_deg(a: list[float], b: list[float], tol: float = 1e-9) -> int:
    def trim(p: list[float]) -> list[float]:
        while len(p) > 1 and abs(p[-1]) < tol:
            p.pop()
        return p

    a, b = trim(list(a)), trim(list(b))
    while any(abs(c) > tol for c in b):
        # polynomial mod
        r = list(a)
        while len(r) >= len(b) and abs(r[-1]) > tol:
            shift = len(r) - len(b)
            coef = r[-1] / b[-1]
            for i, c in enumerate(b):
                r[i + shift] -= coef * c
            r = trim(r)
        a, b = b, trim(r)
    return len(trim(a)) - 1


def _bench_separable_check(seed: int = 0) -> float:
    checks = []
    # f = x^2 - 1: f' = 2x, gcd = 1 -> separable
    checks.append(_poly_gcd_deg([-1.0, 0.0, 1.0], _poly_deriv([-1.0, 0.0, 1.0])) == 0)
    # f = (x-1)^2 = x^2 - 2x + 1: f' shares factor x-1 -> deg gcd = 1
    checks.append(_poly_gcd_deg([1.0, -2.0, 1.0], _poly_deriv([1.0, -2.0, 1.0])) == 1)
    # f = (x-1)^3 -> deg gcd = 2
    checks.append(_poly_gcd_deg([-1.0, 3.0, -3.0, 1.0], _poly_deriv([-1.0, 3.0, -3.0, 1.0])) == 2)
    # f = x^3 - 2 (Q): separable
    checks.append(_poly_gcd_deg([-2.0, 0.0, 0.0, 1.0], _poly_deriv([-2.0, 0.0, 0.0, 1.0])) == 0)
    # f = x^2(x+1) = x^3 + x^2: deg gcd(x, x^3+x^2) -> x | f, gcd deg 1
    checks.append(_poly_gcd_deg([0.0, 0.0, 1.0, 1.0], _poly_deriv([0.0, 0.0, 1.0, 1.0])) == 1)
    # product rule sanity: deg gcd counts common-root multiplicity dim
    checks.append(_poly_gcd_deg([1.0, 0.0, -1.0], [1.0, -1.0]) == 1)  # x-1 shared
    return float(sum(checks) / len(checks))


def bench_separable_check(seed: int = 0) -> dict[str, float]:
    return {"synthetic_separable_check": _bench_separable_check(seed)}
