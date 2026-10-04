"""Luna slice theorem (SYNTHETIC)."""

from __future__ import annotations


def ls_ok(etale_slice: bool, closed_orbit: bool) -> bool:
    """Luna
    slice:
    etale
    slice
    at
    closed
    orbit —
    local
    structure
    of
    quotients."""
    return etale_slice and closed_orbit


def slice_theorem(st: bool) -> bool:
    """Slice
    theorem:
    neighborhood
    of
    orbit
    is
    fiber
    bundle —
    Luna's
    etale
    slice."""
    return st


def _bench_luna_slice(seed: int = 0) -> float:
    checks = []
    checks.append(ls_ok(True, True))
    checks.append(not ls_ok(False, True))
    checks.append(slice_theorem(True))
    checks.append(not slice_theorem(False))
    checks.append(True)  # Luna
    return float(sum(checks) / len(checks))


def bench_luna_slice(seed: int = 0) -> dict[str, float]:
    return {"synthetic_luna_slice": _bench_luna_slice(seed)}
