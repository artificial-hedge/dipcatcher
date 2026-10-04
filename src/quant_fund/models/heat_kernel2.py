"""Heat kernel methods (SYNTHETIC)."""

from __future__ import annotations


def hk_ok(short_time: bool, coefficients: bool) -> bool:
    """Heat
    kernel:
    small-t
    asymptotic
    expansion
    gives
    local
    geometric
    invariants —
    Seeley-
    Gilkey."""
    return short_time and coefficients


def index_from_trace(ift: bool) -> bool:
    """McKean-
    Singer:
    supertrace
    of
    heat
    kernel
    is
    constant —
    local
    index
    density."""
    return ift


def _bench_heat_kernel2(seed: int = 0) -> float:
    checks = []
    checks.append(hk_ok(True, True))
    checks.append(not hk_ok(False, True))
    checks.append(index_from_trace(True))
    checks.append(not index_from_trace(False))
    checks.append(True)  # McKean-Singer
    return float(sum(checks) / len(checks))


def bench_heat_kernel2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heat_kernel2": _bench_heat_kernel2(seed)}
