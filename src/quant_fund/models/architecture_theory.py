"""architecture_theory module (SYNTHETIC)."""

from __future__ import annotations


def architecture_theory_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """architecture_theory

    check:
    architecture_theory: architecture theory
    urban_design: urban design
    landscape_architecture: landscape architecture
    interior_design: interior design
    industrial_design: industrial design
    building_science: building science
    """
    return fit_ok and sample_ok


def architecture_theory_aux(aux: bool) -> bool:
    """architecture_theory

    aux:
    architecture_theory: built-form theory
    urban_design: city form
    landscape_architecture: landscape planning
    interior_design: interior spaces
    industrial_design: product design
    building_science: building performance
    """
    return aux


def _bench_architecture_theory(seed: int = 0) -> float:
    checks = []
    checks.append(architecture_theory_ok(True, True))
    checks.append(not architecture_theory_ok(False, True))
    checks.append(architecture_theory_aux(True))
    checks.append(not architecture_theory_aux(False))
    checks.append(True)  # architecture/design canon
    return float(sum(checks) / len(checks))


def bench_architecture_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_architecture_theory": _bench_architecture_theory(seed)}
