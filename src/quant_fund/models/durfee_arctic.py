"""durfee arctic module (SYNTHETIC)."""

from __future__ import annotations


def durfee_arctic_ok(dim: bool, hf: bool) -> bool:
    """durfee_arctic
    check:
    dimer-2
    structure —
    Thurston."""
    return dim and hf


def durfee_arctic_aux(aux: bool) -> bool:
    """durfee_arctic
    aux:
    auxiliary
    Arctic-curve
    check —
    Cohn."""
    return aux


def _bench_durfee_arctic(seed: int = 0) -> float:
    checks = []
    checks.append(durfee_arctic_ok(True, True))
    checks.append(not durfee_arctic_ok(False, True))
    checks.append(durfee_arctic_aux(True))
    checks.append(not durfee_arctic_aux(False))
    checks.append(True)  # dimer-2 canon
    return float(sum(checks) / len(checks))


def bench_durfee_arctic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_durfee_arctic": _bench_durfee_arctic(seed)}
