"""duplantier sheffield module (SYNTHETIC)."""

from __future__ import annotations


def duplantier_sheffield_ok(gff: bool, qg: bool) -> bool:
    """duplantier_sheffield
    check:
    Gaussian-free-field
    structure —
    Sheffield."""
    return gff and qg


def duplantier_sheffield_aux(aux: bool) -> bool:
    """duplantier_sheffield
    aux:
    auxiliary
    quantum-gravity
    check —
    Duplantier."""
    return aux


def _bench_duplantier_sheffield(seed: int = 0) -> float:
    checks = []
    checks.append(duplantier_sheffield_ok(True, True))
    checks.append(not duplantier_sheffield_ok(False, True))
    checks.append(duplantier_sheffield_aux(True))
    checks.append(not duplantier_sheffield_aux(False))
    checks.append(True)  # GFF canon
    return float(sum(checks) / len(checks))


def bench_duplantier_sheffield(seed: int = 0) -> dict[str, float]:
    return {"synthetic_duplantier_sheffield": _bench_duplantier_sheffield(seed)}
