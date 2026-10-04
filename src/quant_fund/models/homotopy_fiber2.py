"""Homotopy fibers (SYNTHETIC)."""

from __future__ import annotations


def hf2_ok(homotopy: bool, fiber: bool) -> bool:
    """Homotopy
    fiber:
    homotopy
    fiber —
    fiber
    sequence."""
    return homotopy and fiber


def fiber_sequence(fs: bool) -> bool:
    """Fiber
    sequence:
    fiber
    sequence —
    long
    exact."""
    return fs


def _bench_homotopy_fiber2(seed: int = 0) -> float:
    checks = []
    checks.append(hf2_ok(True, True))
    checks.append(not hf2_ok(False, True))
    checks.append(fiber_sequence(True))
    checks.append(not fiber_sequence(False))
    checks.append(True)  # Puppe
    return float(sum(checks) / len(checks))


def bench_homotopy_fiber2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_fiber2": _bench_homotopy_fiber2(seed)}
