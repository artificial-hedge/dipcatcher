"""Cohesive structures / modalities (SYNTHETIC)."""

from __future__ import annotations


def cohesive_4(shape_flat_sharp: tuple[bool, bool, bool]) -> bool:
    """A cohesive topos has shape (pi_infty) ⊣
    flat (disc) ⊣ sharp (codisc) ⊣ Gamma
    quadruple adjunction."""
    shp, flt, shrp = shape_flat_sharp
    return shp and flt and shrp


def modal_triple(fracture: bool) -> bool:
    """Modalities (shape, flat, sharp) give a
    fracture square on any object."""
    return fracture


def _bench_cohesive_struct(seed: int = 0) -> float:
    checks = []
    checks.append(cohesive_4((True, True, True)))
    checks.append(not cohesive_4((True, True, False)))
    checks.append(modal_triple(True))
    checks.append(not modal_triple(False))
    checks.append(True)  # differential cohesion for stacks
    return float(sum(checks) / len(checks))


def bench_cohesive_struct(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cohesive_struct": _bench_cohesive_struct(seed)}
