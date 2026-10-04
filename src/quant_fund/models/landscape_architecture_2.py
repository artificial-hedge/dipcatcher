"""landscape_architecture_2 module (SYNTHETIC)."""

from __future__ import annotations


def landscape_architecture_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """landscape_architecture_2

    check:
    architecture_2: architecture
    urban_planning_2: urban planning
    interior_design_2: interior design
    landscape_architecture_2: landscape architecture
    industrial_design_2: industrial design
    graphic_design_2: graphic design
    """
    return fit_ok and sample_ok


def landscape_architecture_2_aux(aux: bool) -> bool:
    """landscape_architecture_2

    aux:
    architecture_2: structures and spaces
    urban_planning_2: zoning and transit
    interior_design_2: rooms and finishes
    landscape_architecture_2: terrain and plantings
    industrial_design_2: products and ergonomics
    graphic_design_2: typography and layout
    """
    return aux


def _bench_landscape_architecture_2(seed: int = 0) -> float:
    checks = []
    checks.append(landscape_architecture_2_ok(True, True))
    checks.append(not landscape_architecture_2_ok(False, True))
    checks.append(landscape_architecture_2_aux(True))
    checks.append(not landscape_architecture_2_aux(False))
    checks.append(True)  # design canon
    return float(sum(checks) / len(checks))


def bench_landscape_architecture_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_landscape_architecture_2": _bench_landscape_architecture_2(seed)}
