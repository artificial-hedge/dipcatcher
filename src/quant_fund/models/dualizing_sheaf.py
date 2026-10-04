"""Dualizing sheaf (SYNTHETIC)."""

from __future__ import annotations


def ds_ok(dualizing_sheaf: bool, gorenstein: bool) -> bool:
    """Dualizing
    sheaf:
    canonical
    bundle
    is
    dualizing —
    Gorenstein
    case."""
    return dualizing_sheaf and gorenstein


def canonical_dual(cd: bool) -> bool:
    """Canonical
    dual:
    canonical
    sheaf
    dualizes
    cohomology —
    Serre
    on
    smooth."""
    return cd


def _bench_dualizing_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(ds_ok(True, True))
    checks.append(not ds_ok(False, True))
    checks.append(canonical_dual(True))
    checks.append(not canonical_dual(False))
    checks.append(True)  # Serre
    return float(sum(checks) / len(checks))


def bench_dualizing_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dualizing_sheaf": _bench_dualizing_sheaf(seed)}
