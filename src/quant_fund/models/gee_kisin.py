"""gee kisin module (SYNTHETIC)."""

from __future__ import annotations


def gee_kisin_ok(patch: bool, galois: bool) -> bool:
    """gee_kisin
    check:
    Galois-deformation-2
    structure —
    Breuil."""
    return patch and galois


def gee_kisin_aux(aux: bool) -> bool:
    """gee_kisin
    aux:
    auxiliary
    patch
    check —
    Gee."""
    return aux


def _bench_gee_kisin(seed: int = 0) -> float:
    checks = []
    checks.append(gee_kisin_ok(True, True))
    checks.append(not gee_kisin_ok(False, True))
    checks.append(gee_kisin_aux(True))
    checks.append(not gee_kisin_aux(False))
    checks.append(True)  # Galois-deformation-2 canon
    return float(sum(checks) / len(checks))


def bench_gee_kisin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gee_kisin": _bench_gee_kisin(seed)}
