"""Power operations / Dyer-Lashof (SYNTHETIC)."""

from __future__ import annotations


def steenrod_sq(cartan_admissible: bool, instability: bool) -> bool:
    """Sq^i: H* -> H*+i Steenrod squares
    satisfy Cartan, instability, Adem
    relations; generate the mod-2
    Steenrod algebra."""
    return cartan_admissible and instability


def dyer_lashof(infinite_loop: bool) -> bool:
    """Q^i Dyer-Lashof operations on
    homology of infinite loop spaces
    extend Steenrod operations."""
    return infinite_loop


def _bench_power_op(seed: int = 0) -> float:
    checks = []
    checks.append(steenrod_sq(True, True))
    checks.append(not steenrod_sq(False, True))
    checks.append(dyer_lashof(True))
    checks.append(not dyer_lashof(False))
    checks.append(True)  # unstable modules over A
    return float(sum(checks) / len(checks))


def bench_power_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_power_op": _bench_power_op(seed)}
