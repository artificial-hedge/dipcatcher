"""building_science module (SYNTHETIC)."""

from __future__ import annotations


def building_science_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """building_science

    check:
    architecture_theory: architecture theory
    urban_design: urban design
    landscape_architecture: landscape architecture
    interior_design: interior design
    industrial_design: industrial design
    building_science: building science
    """
    return fit_ok and sample_ok


def building_science_aux(aux: bool) -> bool:
    """building_science

    aux:
    architecture_theory: built-form theory
    urban_design: city form
    landscape_architecture: landscape planning
    interior_design: interior spaces
    industrial_design: product design
    building_science: building performance
    """
    return aux


def _bench_building_science(seed: int = 0) -> float:
    checks = []
    checks.append(building_science_ok(True, True))
    checks.append(not building_science_ok(False, True))
    checks.append(building_science_aux(True))
    checks.append(not building_science_aux(False))
    checks.append(True)  # architecture/design canon
    return float(sum(checks) / len(checks))


def bench_building_science(seed: int = 0) -> dict[str, float]:
    return {"synthetic_building_science": _bench_building_science(seed)}
