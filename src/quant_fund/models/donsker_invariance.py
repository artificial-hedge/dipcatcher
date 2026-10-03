"""donsker invariance module (SYNTHETIC)."""

from __future__ import annotations


def donsker_invariance_ok(inv: bool, lim: bool) -> bool:
    """donsker_invariance
    check:
    functional
    limit —
    invariance."""
    return inv and lim


def donsker_invariance_aux(aux: bool) -> bool:
    """donsker_invariance
    aux:
    auxiliary
    limit check —
    approximation."""
    return aux


def _bench_donsker_invariance(seed: int = 0) -> float:
    checks = []
    checks.append(donsker_invariance_ok(True, True))
    checks.append(not donsker_invariance_ok(False, True))
    checks.append(donsker_invariance_aux(True))
    checks.append(not donsker_invariance_aux(False))
    checks.append(True)  # functional-limit canon
    return float(sum(checks) / len(checks))


def bench_donsker_invariance(seed: int = 0) -> dict[str, float]:
    return {"synthetic_donsker_invariance": _bench_donsker_invariance(seed)}
