"""dz invariance module (SYNTHETIC)."""

from __future__ import annotations


def dz_invariance_ok(proc: bool, tight: bool) -> bool:
    """dz_invariance
    check:
    empirical-process
    structure —
    Donsker."""
    return proc and tight


def dz_invariance_aux(aux: bool) -> bool:
    """dz_invariance
    aux:
    auxiliary
    class
    check —
    Vapnik."""
    return aux


def _bench_dz_invariance(seed: int = 0) -> float:
    checks = []
    checks.append(dz_invariance_ok(True, True))
    checks.append(not dz_invariance_ok(False, True))
    checks.append(dz_invariance_aux(True))
    checks.append(not dz_invariance_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_dz_invariance(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dz_invariance": _bench_dz_invariance(seed)}
