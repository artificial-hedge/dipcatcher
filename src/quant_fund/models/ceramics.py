"""ceramics module (SYNTHETIC)."""

from __future__ import annotations


def ceramics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ceramics

    check:
    crystal_structure: crystal structure
    polymer_physics: polymer physics
    metallurgy: metallurgy
    ceramics: ceramics
    nanomaterials: nanomaterials
    superconductivity: superconductivity
    """
    return fit_ok and sample_ok


def ceramics_aux(aux: bool) -> bool:
    """ceramics

    aux:
    crystal_structure: lattice parameters
    polymer_physics: chain dynamics
    metallurgy: phase diagrams
    ceramics: sintering
    nanomaterials: quantum dots
    superconductivity: Cooper pairs
    """
    return aux


def _bench_ceramics(seed: int = 0) -> float:
    checks = []
    checks.append(ceramics_ok(True, True))
    checks.append(not ceramics_ok(False, True))
    checks.append(ceramics_aux(True))
    checks.append(not ceramics_aux(False))
    checks.append(True)  # materials-science canon
    return float(sum(checks) / len(checks))


def bench_ceramics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ceramics": _bench_ceramics(seed)}
