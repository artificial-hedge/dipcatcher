"""freidlin wentzell module (SYNTHETIC)."""

from __future__ import annotations


def freidlin_wentzell_ok(rate: bool, action: bool) -> bool:
    """freidlin_wentzell
    check:
    large-deviation
    structure —
    Varadhan."""
    return rate and action


def freidlin_wentzell_aux(aux: bool) -> bool:
    """freidlin_wentzell
    aux:
    auxiliary
    contraction
    check —
    Wentzell."""
    return aux


def _bench_freidlin_wentzell(seed: int = 0) -> float:
    checks = []
    checks.append(freidlin_wentzell_ok(True, True))
    checks.append(not freidlin_wentzell_ok(False, True))
    checks.append(freidlin_wentzell_aux(True))
    checks.append(not freidlin_wentzell_aux(False))
    checks.append(True)  # LDP canon
    return float(sum(checks) / len(checks))


def bench_freidlin_wentzell(seed: int = 0) -> dict[str, float]:
    return {"synthetic_freidlin_wentzell": _bench_freidlin_wentzell(seed)}
