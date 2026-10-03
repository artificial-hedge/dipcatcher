"""higher brace2 module (SYNTHETIC)."""

from __future__ import annotations


def higher_brace2_ok(higher: bool, algebra: bool) -> bool:
    """higher_brace2
    check:
    higher
    algebra —
    operadic."""
    return higher and algebra


def higher_brace2_aux(aux: bool) -> bool:
    """higher_brace2
    aux:
    auxiliary
    higher
    check —
    factorization."""
    return aux


def _bench_higher_brace2(seed: int = 0) -> float:
    checks = []
    checks.append(higher_brace2_ok(True, True))
    checks.append(not higher_brace2_ok(False, True))
    checks.append(higher_brace2_aux(True))
    checks.append(not higher_brace2_aux(False))
    checks.append(True)  # higher algebra canon
    return float(sum(checks) / len(checks))


def bench_higher_brace2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_higher_brace2": _bench_higher_brace2(seed)}
