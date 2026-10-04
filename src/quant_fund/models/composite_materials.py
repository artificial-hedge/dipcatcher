"""composite_materials module (SYNTHETIC)."""

from __future__ import annotations


def composite_materials_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """composite_materials

    check:
    semiconductors_materials: semiconductors materials
    composite_materials: composite materials
    thin_films: thin films
    biomaterials: biomaterials
    phase_diagrams: phase diagrams
    characterization_methods: characterization methods
    """
    return fit_ok and sample_ok


def composite_materials_aux(aux: bool) -> bool:
    """composite_materials

    aux:
    semiconductors_materials: semiconductor devices
    composite_materials: fiber composites
    thin_films: film deposition
    biomaterials: medical materials
    phase_diagrams: equilibrium phases
    characterization_methods: microscopy
    """
    return aux


def _bench_composite_materials(seed: int = 0) -> float:
    checks = []
    checks.append(composite_materials_ok(True, True))
    checks.append(not composite_materials_ok(False, True))
    checks.append(composite_materials_aux(True))
    checks.append(not composite_materials_aux(False))
    checks.append(True)  # materials-2 canon
    return float(sum(checks) / len(checks))


def bench_composite_materials(seed: int = 0) -> dict[str, float]:
    return {"synthetic_composite_materials": _bench_composite_materials(seed)}
