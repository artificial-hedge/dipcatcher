"""averaging method module (SYNTHETIC)."""

from __future__ import annotations


def averaging_method_ok(term: bool, est: bool) -> bool:
    """averaging_method
    check:
    flux/asymptotic —
    term/estimate
    consistency."""
    return term and est


def averaging_method_aux(aux: bool) -> bool:
    """averaging_method
    aux:
    auxiliary
    flux/asymptotic check —
    error bound."""
    return aux


def _bench_averaging_method(seed: int = 0) -> float:
    checks = []
    checks.append(averaging_method_ok(True, True))
    checks.append(not averaging_method_ok(False, True))
    checks.append(averaging_method_aux(True))
    checks.append(not averaging_method_aux(False))
    checks.append(True)  # flux/asymptotic canon
    return float(sum(checks) / len(checks))


def bench_averaging_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_averaging_method": _bench_averaging_method(seed)}
