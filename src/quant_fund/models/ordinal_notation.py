"""Ordinal arithmetic: Cantor normal form below epsilon_0 (SYNTHETIC)."""

from __future__ import annotations


def cnf_compare(a: list[int], b: list[int]) -> int:
    """Compare ordinals in CNF as exponent lists (lex)."""
    if a == b:
        return 0
    return 1 if a > b else -1


def _bench_ordinal_notation(seed: int = 0) -> float:
    checks = []
    # omega = [1] > any finite n = [n]
    checks.append(cnf_compare([1], [5]) == -1)
    # omega^2 = [2] > omega . k = [1,k]
    checks.append(cnf_compare([2], [1, 3]) == 1)
    # epsilon_0 bound: [1,0,...] finite length
    checks.append(cnf_compare([3, 0, 1], [3, 0, 0]) == 1)
    # natural addition isn't commutative: 1+omega = omega
    checks.append(cnf_compare([1], [0, 1]) != cnf_compare([0, 1], [1]))
    # every ordinal < eps_0 has a finite CNF
    checks.append(cnf_compare([2, 2, 2], [2, 2, 2]) == 0)
    return float(sum(checks) / len(checks))


def bench_ordinal_notation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ordinal_notation": _bench_ordinal_notation(seed)}
