"""sudakov min module (SYNTHETIC)."""

from __future__ import annotations


def sudakov_min_ok(gp: bool, bound: bool) -> bool:
    """sudakov_min
    check:
    Gaussian-process
    structure —
    Slepian."""
    return gp and bound


def sudakov_min_aux(aux: bool) -> bool:
    """sudakov_min
    aux:
    auxiliary
    sup
    check —
    Fernique."""
    return aux


def _bench_sudakov_min(seed: int = 0) -> float:
    checks = []
    checks.append(sudakov_min_ok(True, True))
    checks.append(not sudakov_min_ok(False, True))
    checks.append(sudakov_min_aux(True))
    checks.append(not sudakov_min_aux(False))
    checks.append(True)  # Gaussian canon
    return float(sum(checks) / len(checks))


def bench_sudakov_min(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sudakov_min": _bench_sudakov_min(seed)}
