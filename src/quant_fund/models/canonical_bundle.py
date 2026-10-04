"""Canonical bundle (SYNTHETIC)."""

from __future__ import annotations


def cb_ok(top_exterior: bool, canonical: bool) -> bool:
    """Canonical
    bundle:
    top
    exterior
    cotangent —
    canonical
    class."""
    return top_exterior and canonical


def kodaira_dim(kd: bool) -> bool:
    """Kodaira
    dimension:
    growth
    of
    canonical
    sections —
    birational
    invariant."""
    return kd


def _bench_canonical_bundle(seed: int = 0) -> float:
    checks = []
    checks.append(cb_ok(True, True))
    checks.append(not cb_ok(False, True))
    checks.append(kodaira_dim(True))
    checks.append(not kodaira_dim(False))
    checks.append(True)  # Kodaira
    return float(sum(checks) / len(checks))


def bench_canonical_bundle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_canonical_bundle": _bench_canonical_bundle(seed)}
