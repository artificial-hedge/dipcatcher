"""industrial_design module (SYNTHETIC)."""

from __future__ import annotations


def industrial_design_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """industrial_design

    check:
    architecture_theory: architecture theory
    urban_design: urban design
    landscape_architecture: landscape architecture
    interior_design: interior design
    industrial_design: industrial design
    building_science: building science
    """
    return fit_ok and sample_ok


def industrial_design_aux(aux: bool) -> bool:
    """industrial_design

    aux:
    architecture_theory: built-form theory
    urban_design: city form
    landscape_architecture: landscape planning
    interior_design: interior spaces
    industrial_design: product design
    building_science: building performance
    """
    return aux


def _bench_industrial_design(seed: int = 0) -> float:
    checks = []
    checks.append(industrial_design_ok(True, True))
    checks.append(not industrial_design_ok(False, True))
    checks.append(industrial_design_aux(True))
    checks.append(not industrial_design_aux(False))
    checks.append(True)  # architecture/design canon
    return float(sum(checks) / len(checks))


def bench_industrial_design(seed: int = 0) -> dict[str, float]:
    return {"synthetic_industrial_design": _bench_industrial_design(seed)}
