"""urban_planning_2 module (SYNTHETIC)."""

from __future__ import annotations


def urban_planning_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """urban_planning_2

    check:
    architecture_2: architecture
    urban_planning_2: urban planning
    interior_design_2: interior design
    landscape_architecture_2: landscape architecture
    industrial_design_2: industrial design
    graphic_design_2: graphic design
    """
    return fit_ok and sample_ok


def urban_planning_2_aux(aux: bool) -> bool:
    """urban_planning_2

    aux:
    architecture_2: structures and spaces
    urban_planning_2: zoning and transit
    interior_design_2: rooms and finishes
    landscape_architecture_2: terrain and plantings
    industrial_design_2: products and ergonomics
    graphic_design_2: typography and layout
    """
    return aux


def _bench_urban_planning_2(seed: int = 0) -> float:
    checks = []
    checks.append(urban_planning_2_ok(True, True))
    checks.append(not urban_planning_2_ok(False, True))
    checks.append(urban_planning_2_aux(True))
    checks.append(not urban_planning_2_aux(False))
    checks.append(True)  # design canon
    return float(sum(checks) / len(checks))


def bench_urban_planning_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_urban_planning_2": _bench_urban_planning_2(seed)}
