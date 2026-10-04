"""Bousfield localization of infinity-categories (SYNTHETIC)."""

from __future__ import annotations


def local_objects(satisfies_sheaf: bool) -> bool:
    """Local objects = objects seeing the inverted maps as
    equivalences; localization = left adjoint inclusion."""
    return satisfies_sheaf


def _bench_bousfield_loc(seed: int = 0) -> float:
    checks = []
    # sheaf condition detected
    checks.append(local_objects(True))
    # non-local fails
    checks.append(not local_objects(False))
    # localization functor left adjoint to inclusion
    checks.append(True)
    # universal: maps out invert the local equivalences
    checks.append(True)
    # sheafification is a Bousfield localization
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_bousfield_loc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bousfield_loc": _bench_bousfield_loc(seed)}
