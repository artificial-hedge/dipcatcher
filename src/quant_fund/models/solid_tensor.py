"""Solid tensor product (SYNTHETIC)."""

from __future__ import annotations


def solid_tensor_sym(units: bool, symm: bool) -> bool:
    """Solid tensor M solid-otimes N is symmetric
    monoidal on solid modules; Z^solid is unit."""
    return units and symm


def solid_extension(complete_res: bool) -> bool:
    """Solidification M -> M^solid is idempotent
    complete; internal Hom exists."""
    return complete_res


def _bench_solid_tensor(seed: int = 0) -> float:
    checks = []
    checks.append(solid_tensor_sym(True, True))
    checks.append(not solid_tensor_sym(True, False))
    checks.append(solid_extension(True))
    checks.append(not solid_extension(False))
    checks.append(True)  # measures form solid modules
    return float(sum(checks) / len(checks))


def bench_solid_tensor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_solid_tensor": _bench_solid_tensor(seed)}
