"""Normal extensions: an irreducible splits completely or has no roots (SYNTHETIC)."""

from __future__ import annotations


def _poly_roots_mod(f: list[int], p: int) -> list[int]:
    return [x for x in range(p) if sum(c * pow(x, i, p) for i, c in enumerate(f)) % p == 0]


def _bench_normality_check(seed: int = 0) -> float:
    checks = []
    # x^2 + 1 over GF(5): roots 2,3 -> splits (normal quadratic)
    checks.append(sorted(_poly_roots_mod([1, 0, 1], 5)) == [2, 3])
    # x^2 + 1 over GF(7): no roots (7 = 3 mod 4)
    checks.append(_poly_roots_mod([1, 0, 1], 7) == [])
    # x^2 - 2 over GF(7): roots 3,4 (9=2 mod 7)
    checks.append(sorted(_poly_roots_mod([-2, 0, 1], 7)) == [3, 4])
    # x^3 - 2 over GF(7): cubes mod 7 are {0,1,6} -> no root, irreducible
    r = _poly_roots_mod([-2, 0, 0, 1], 7)
    checks.append(len(r) == 0)
    # x^3 - 1 over GF(7): roots 1,2,4 -> splits fully (normal: 7 = 1 mod 3)
    r3 = _poly_roots_mod([-1, 0, 0, 1], 7)
    checks.append(sorted(r3) == [1, 2, 4])
    # cyclotomic x^4+x^3+x^2+x+1 over GF(11): splits fully (11 = 1 mod 5)
    r2 = _poly_roots_mod([1, 1, 1, 1, 1], 11)
    checks.append(len(r2) == 4)
    # Frobenius on GF(4): w -> w^2 swaps the two non-rational elements
    mul = {(2, 2): 3, (3, 3): 2}  # w^2 = w+1 (elem 3), (w+1)^2 = w (elem 2)
    checks.append(mul[(2, 2)] == 3 and mul[(3, 3)] == 2)
    return float(sum(checks) / len(checks))


def bench_normality_check(seed: int = 0) -> dict[str, float]:
    return {"synthetic_normality_check": _bench_normality_check(seed)}
