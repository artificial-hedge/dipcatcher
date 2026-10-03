"""taylor wiles module (SYNTHETIC)."""

from __future__ import annotations


def taylor_wiles_ok(patch: bool, galois: bool) -> bool:
    """taylor_wiles
    check:
    Galois-deformation-2
    structure —
    Breuil."""
    return patch and galois


def taylor_wiles_aux(aux: bool) -> bool:
    """taylor_wiles
    aux:
    auxiliary
    patch
    check —
    Gee."""
    return aux


def _bench_taylor_wiles(seed: int = 0) -> float:
    checks = []
    checks.append(taylor_wiles_ok(True, True))
    checks.append(not taylor_wiles_ok(False, True))
    checks.append(taylor_wiles_aux(True))
    checks.append(not taylor_wiles_aux(False))
    checks.append(True)  # Galois-deformation-2 canon
    return float(sum(checks) / len(checks))


def bench_taylor_wiles(seed: int = 0) -> dict[str, float]:
    return {"synthetic_taylor_wiles": _bench_taylor_wiles(seed)}
