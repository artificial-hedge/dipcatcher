"""characterization_methods module (SYNTHETIC)."""

from __future__ import annotations


def characterization_methods_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """characterization_methods

    check:
    semiconductors_materials: semiconductors materials
    composite_materials: composite materials
    thin_films: thin films
    biomaterials: biomaterials
    phase_diagrams: phase diagrams
    characterization_methods: characterization methods
    """
    return fit_ok and sample_ok


def characterization_methods_aux(aux: bool) -> bool:
    """characterization_methods

    aux:
    semiconductors_materials: semiconductor devices
    composite_materials: fiber composites
    thin_films: film deposition
    biomaterials: medical materials
    phase_diagrams: equilibrium phases
    characterization_methods: microscopy
    """
    return aux


def _bench_characterization_methods(seed: int = 0) -> float:
    checks = []
    checks.append(characterization_methods_ok(True, True))
    checks.append(not characterization_methods_ok(False, True))
    checks.append(characterization_methods_aux(True))
    checks.append(not characterization_methods_aux(False))
    checks.append(True)  # materials-2 canon
    return float(sum(checks) / len(checks))


def bench_characterization_methods(seed: int = 0) -> dict[str, float]:
    return {"synthetic_characterization_methods": _bench_characterization_methods(seed)}
