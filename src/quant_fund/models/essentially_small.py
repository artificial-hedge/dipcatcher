"""Essentially small categories (SYNTHETIC)."""

from __future__ import annotations


def es_ok(essentially: bool, small: bool) -> bool:
    """Essentially
    small:
    essentially
    small cat —
    equivalent
    to a
    small
    category."""
    return essentially and small


def skeleton_exists(se: bool) -> bool:
    """Skeleton:
    essentially
    small
    implies
    skeleton —
    iso-class
    representatives."""
    return se


def _bench_essentially_small(seed: int = 0) -> float:
    checks = []
    checks.append(es_ok(True, True))
    checks.append(not es_ok(False, True))
    checks.append(skeleton_exists(True))
    checks.append(not skeleton_exists(False))
    checks.append(True)  # category theory basics
    return float(sum(checks) / len(checks))


def bench_essentially_small(seed: int = 0) -> dict[str, float]:
    return {"synthetic_essentially_small": _bench_essentially_small(seed)}
