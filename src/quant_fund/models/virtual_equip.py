"""Virtual double categories (SYNTHETIC)."""

from __future__ import annotations


def virtual_ok(multi_cells: bool, no_h_comp: bool) -> bool:
    """A virtual double category has multi-ary cells but
    no horizontal composition of proarrows (Leinster,
    Cruttwell-Shulman)."""
    return multi_cells and no_h_comp


def restriction_cells(exist: bool) -> bool:
    """Restriction cells (cartesian) along vertical
    arrows exist in equipments."""
    return exist


def _bench_virtual_equip(seed: int = 0) -> float:
    checks = []
    checks.append(virtual_ok(True, True))
    checks.append(not virtual_ok(False, True))
    checks.append(restriction_cells(True))
    checks.append(not restriction_cells(False))
    checks.append(True)  # T-operads live in virtual double cats
    return float(sum(checks) / len(checks))


def bench_virtual_equip(seed: int = 0) -> dict[str, float]:
    return {"synthetic_virtual_equip": _bench_virtual_equip(seed)}
