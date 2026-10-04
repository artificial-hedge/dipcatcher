"""Tate object in motives (SYNTHETIC)."""

from __future__ import annotations


def to_ok(tate: bool, object_motive: bool) -> bool:
    """Tate:
    Tate
    motive
    Z(1) —
    Tate
    object."""
    return tate and object_motive


def tate_suspension(ts: bool) -> bool:
    """Tate
    suspension:
    Tate
    twisting
    (n)
    suspensions —
    Tate
    twist."""
    return ts


def _bench_tate_object(seed: int = 0) -> float:
    checks = []
    checks.append(to_ok(True, True))
    checks.append(not to_ok(False, True))
    checks.append(tate_suspension(True))
    checks.append(not tate_suspension(False))
    checks.append(True)  # Tate
    return float(sum(checks) / len(checks))


def bench_tate_object(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tate_object": _bench_tate_object(seed)}
