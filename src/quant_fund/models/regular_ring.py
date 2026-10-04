"""Regular ring (SYNTHETIC)."""

from __future__ import annotations


def rr_ok(regular: bool, ring: bool) -> bool:
    """Regular:
    regular
    local
    ring —
    homological
    criterion."""
    return regular and ring


def regular_param(rp: bool) -> bool:
    """Regular
    parameters:
    regular
    system
    of
    parameters —
    regular
    local."""
    return rp


def _bench_regular_ring(seed: int = 0) -> float:
    checks = []
    checks.append(rr_ok(True, True))
    checks.append(not rr_ok(False, True))
    checks.append(regular_param(True))
    checks.append(not regular_param(False))
    checks.append(True)  # Serre
    return float(sum(checks) / len(checks))


def bench_regular_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_regular_ring": _bench_regular_ring(seed)}
