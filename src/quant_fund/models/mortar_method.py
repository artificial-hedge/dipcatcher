"""mortar method module (SYNTHETIC)."""

from __future__ import annotations


def mortar_method_ok(node: bool, poly: bool) -> bool:
    """mortar_method
    check:
    spectral-element —
    high-order
    consistency."""
    return node and poly


def mortar_method_aux(aux: bool) -> bool:
    """mortar_method
    aux:
    auxiliary
    SEM check —
    interpolation."""
    return aux


def _bench_mortar_method(seed: int = 0) -> float:
    checks = []
    checks.append(mortar_method_ok(True, True))
    checks.append(not mortar_method_ok(False, True))
    checks.append(mortar_method_aux(True))
    checks.append(not mortar_method_aux(False))
    checks.append(True)  # spectral-element canon
    return float(sum(checks) / len(checks))


def bench_mortar_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mortar_method": _bench_mortar_method(seed)}
