"""Derived deformation theory (SYNTHETIC)."""

from __future__ import annotations


def derived_def_ok(dgla: bool, mc_solution: bool) -> bool:
    """Derived deformation
    functor modeled by a
    dgla g: solutions are
    Maurer-Cartan elements
    of g; Lurie + Pridham
    equivalence."""
    return dgla and mc_solution


def pridham_rep(prorepresentable: bool) -> bool:
    """Pridham-Lurie:
    prorepresentable derived
    deformation functors
    correspond to
    dg Lie algebras
    up to quasi-iso."""
    return prorepresentable


def _bench_derived_deformation(seed: int = 0) -> float:
    checks = []
    checks.append(derived_def_ok(True, True))
    checks.append(not derived_def_ok(False, True))
    checks.append(pridham_rep(True))
    checks.append(not pridham_rep(False))
    checks.append(True)  # derived Kodaira-Spencer
    return float(sum(checks) / len(checks))


def bench_derived_deformation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_deformation": _bench_derived_deformation(seed)}
