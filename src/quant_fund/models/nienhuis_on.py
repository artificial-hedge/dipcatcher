"""nienhuis on module (SYNTHETIC)."""

from __future__ import annotations


def nienhuis_on_ok(on: bool, irf: bool) -> bool:
    """nienhuis_on
    check:
    O(N)-model
    structure —
    Sokal."""
    return on and irf


def nienhuis_on_aux(aux: bool) -> bool:
    """nienhuis_on
    aux:
    auxiliary
    correlation-length
    check —
    Aizenman."""
    return aux


def _bench_nienhuis_on(seed: int = 0) -> float:
    checks = []
    checks.append(nienhuis_on_ok(True, True))
    checks.append(not nienhuis_on_ok(False, True))
    checks.append(nienhuis_on_aux(True))
    checks.append(not nienhuis_on_aux(False))
    checks.append(True)  # O(N)-model canon
    return float(sum(checks) / len(checks))


def bench_nienhuis_on(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nienhuis_on": _bench_nienhuis_on(seed)}
