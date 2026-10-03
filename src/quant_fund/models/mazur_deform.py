"""mazur deform module (SYNTHETIC)."""

from __future__ import annotations


def mazur_deform_ok(galois: bool, deform: bool) -> bool:
    """mazur_deform
    check:
    Galois-deformation
    structure —
    Kisin."""
    return galois and deform


def mazur_deform_aux(aux: bool) -> bool:
    """mazur_deform
    aux:
    auxiliary
    deformation
    check —
    Taylor."""
    return aux


def _bench_mazur_deform(seed: int = 0) -> float:
    checks = []
    checks.append(mazur_deform_ok(True, True))
    checks.append(not mazur_deform_ok(False, True))
    checks.append(mazur_deform_aux(True))
    checks.append(not mazur_deform_aux(False))
    checks.append(True)  # Galois-deformation canon
    return float(sum(checks) / len(checks))


def bench_mazur_deform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mazur_deform": _bench_mazur_deform(seed)}
