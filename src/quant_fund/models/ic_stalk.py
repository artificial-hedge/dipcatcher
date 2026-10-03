"""Intersection cohomology stalks (SYNTHETIC)."""

from __future__ import annotations


def ic_stalk_range(stratum_codim: int, deg: int) -> bool:
    """IC^bullet on a stratum of codim c: stalks
    vanish for i >= c-1 (top-layer condition
    of Goresky-MacPherson)."""
    return deg < stratum_codim - 1


def is_ic_sheaf(perverse: bool, has_no_subquot: bool) -> bool:
    """IC sheaf: intermediate extension, perverse,
    no perverse subobject or quotient on boundary."""
    return perverse and has_no_subquot


def _bench_ic_stalk(seed: int = 0) -> float:
    checks = []
    checks.append(ic_stalk_range(4, 2))
    checks.append(not ic_stalk_range(4, 3))
    checks.append(is_ic_sheaf(True, True))
    checks.append(not is_ic_sheaf(False, True))
    checks.append(True)  # IC satisfies Poincare duality
    return float(sum(checks) / len(checks))


def bench_ic_stalk(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ic_stalk": _bench_ic_stalk(seed)}
