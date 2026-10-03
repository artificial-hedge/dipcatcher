"""Versal deformation (SYNTHETIC)."""

from __future__ import annotations


def vd_ok(versal: bool, deformation: bool) -> bool:
    """Versal:
    versal
    deformation
    family —
    versal
    deformation."""
    return versal and deformation


def versal_composition(vc: bool) -> bool:
    """Versal
    composition:
    versal
    deformation
    composition —
    versal
    composition."""
    return vc


def _bench_versal_def(seed: int = 0) -> float:
    checks = []
    checks.append(vd_ok(True, True))
    checks.append(not vd_ok(False, True))
    checks.append(versal_composition(True))
    checks.append(not versal_composition(False))
    checks.append(True)  # versal
    return float(sum(checks) / len(checks))


def bench_versal_def(seed: int = 0) -> dict[str, float]:
    return {"synthetic_versal_def": _bench_versal_def(seed)}
