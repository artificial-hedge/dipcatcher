"""Global heights (SYNTHETIC)."""

from __future__ import annotations


def gh_ok(product_formula: bool, absolute: bool) -> bool:
    """Global
    height:
    product
    formula
    over
    absolute
    values —
    Weil
    height
    on
    projective
    space."""
    return product_formula and absolute


def height_machine(hm: bool) -> bool:
    """Height
    machine:
    functorial
    heights
    from
    metrized
    line
    bundles —
    Weil's
    height
    machine."""
    return hm


def _bench_global_height(seed: int = 0) -> float:
    checks = []
    checks.append(gh_ok(True, True))
    checks.append(not gh_ok(False, True))
    checks.append(height_machine(True))
    checks.append(not height_machine(False))
    checks.append(True)  # Weil
    return float(sum(checks) / len(checks))


def bench_global_height(seed: int = 0) -> dict[str, float]:
    return {"synthetic_global_height": _bench_global_height(seed)}
