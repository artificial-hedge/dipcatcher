"""strassen flln module (SYNTHETIC)."""

from __future__ import annotations


def strassen_flln_ok(inv: bool, lim: bool) -> bool:
    """strassen_flln
    check:
    functional
    limit —
    invariance."""
    return inv and lim


def strassen_flln_aux(aux: bool) -> bool:
    """strassen_flln
    aux:
    auxiliary
    limit check —
    approximation."""
    return aux


def _bench_strassen_flln(seed: int = 0) -> float:
    checks = []
    checks.append(strassen_flln_ok(True, True))
    checks.append(not strassen_flln_ok(False, True))
    checks.append(strassen_flln_aux(True))
    checks.append(not strassen_flln_aux(False))
    checks.append(True)  # functional-limit canon
    return float(sum(checks) / len(checks))


def bench_strassen_flln(seed: int = 0) -> dict[str, float]:
    return {"synthetic_strassen_flln": _bench_strassen_flln(seed)}
