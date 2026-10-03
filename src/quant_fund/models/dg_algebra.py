"""dg algebra (SYNTHETIC)."""

from __future__ import annotations


def da_ok(dg: bool, algebra: bool) -> bool:
    """dg
    algebra:
    differential
    graded
    algebra —
    Sullivan."""
    return dg and algebra


def sullivan_model(sm: bool) -> bool:
    """Sullivan:
    minimal
    Sullivan
    dg
    model —
    Sullivan
    model."""
    return sm


def _bench_dg_algebra(seed: int = 0) -> float:
    checks = []
    checks.append(da_ok(True, True))
    checks.append(not da_ok(False, True))
    checks.append(sullivan_model(True))
    checks.append(not sullivan_model(False))
    checks.append(True)  # Sullivan
    return float(sum(checks) / len(checks))


def bench_dg_algebra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dg_algebra": _bench_dg_algebra(seed)}
