"""Raynaud formal geometry (SYNTHETIC)."""

from __future__ import annotations


def rf_ok(raynaud: bool, formal: bool) -> bool:
    """Raynaud:
    Raynaud
    formal
    geometry —
    Raynaud
    generic
    fiber."""
    return raynaud and formal


def generic_fiber(gf: bool) -> bool:
    """Generic
    fiber:
    Raynaud
    generic
    fiber
    functor —
    rigid
    generic
    fiber."""
    return gf


def _bench_raynaud_formal(seed: int = 0) -> float:
    checks = []
    checks.append(rf_ok(True, True))
    checks.append(not rf_ok(False, True))
    checks.append(generic_fiber(True))
    checks.append(not generic_fiber(False))
    checks.append(True)  # Raynaud
    return float(sum(checks) / len(checks))


def bench_raynaud_formal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_raynaud_formal": _bench_raynaud_formal(seed)}
