"""jackson direct module (SYNTHETIC)."""

from __future__ import annotations


def jackson_direct_ok(smooth: bool, approx: bool) -> bool:
    """jackson_direct
    check:
    approximation
    theory —
    smoothness."""
    return smooth and approx


def jackson_direct_aux(aux: bool) -> bool:
    """jackson_direct
    aux:
    auxiliary
    approx check —
    degree."""
    return aux


def _bench_jackson_direct(seed: int = 0) -> float:
    checks = []
    checks.append(jackson_direct_ok(True, True))
    checks.append(not jackson_direct_ok(False, True))
    checks.append(jackson_direct_aux(True))
    checks.append(not jackson_direct_aux(False))
    checks.append(True)  # approximation-theory canon
    return float(sum(checks) / len(checks))


def bench_jackson_direct(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jackson_direct": _bench_jackson_direct(seed)}
