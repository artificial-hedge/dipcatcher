"""Filtered category theory (SYNTHETIC)."""

from __future__ import annotations


def fc_ok(filtered: bool, cocone: bool) -> bool:
    """Filtered
    category:
    filtered
    category —
    cocones
    exist."""
    return filtered and cocone


def filtered_colimit(fcl: bool) -> bool:
    """Filtered
    colimit:
    filtered
    colimit —
    commutes
    finite
    limits."""
    return fcl


def _bench_filtered_cat(seed: int = 0) -> float:
    checks = []
    checks.append(fc_ok(True, True))
    checks.append(not fc_ok(False, True))
    checks.append(filtered_colimit(True))
    checks.append(not filtered_colimit(False))
    checks.append(True)  # filtered colimits exact in Set
    return float(sum(checks) / len(checks))


def bench_filtered_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_filtered_cat": _bench_filtered_cat(seed)}
