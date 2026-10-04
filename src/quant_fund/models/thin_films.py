"""thin_films module (SYNTHETIC)."""

from __future__ import annotations


def thin_films_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """thin_films

    check:
    semiconductors_materials: semiconductors materials
    composite_materials: composite materials
    thin_films: thin films
    biomaterials: biomaterials
    phase_diagrams: phase diagrams
    characterization_methods: characterization methods
    """
    return fit_ok and sample_ok


def thin_films_aux(aux: bool) -> bool:
    """thin_films

    aux:
    semiconductors_materials: semiconductor devices
    composite_materials: fiber composites
    thin_films: film deposition
    biomaterials: medical materials
    phase_diagrams: equilibrium phases
    characterization_methods: microscopy
    """
    return aux


def _bench_thin_films(seed: int = 0) -> float:
    checks = []
    checks.append(thin_films_ok(True, True))
    checks.append(not thin_films_ok(False, True))
    checks.append(thin_films_aux(True))
    checks.append(not thin_films_aux(False))
    checks.append(True)  # materials-2 canon
    return float(sum(checks) / len(checks))


def bench_thin_films(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thin_films": _bench_thin_films(seed)}
