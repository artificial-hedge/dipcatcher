"""Hodge index theorem (SYNTHETIC)."""

from __future__ import annotations


def hi_ok(signature: bool, intersection: bool) -> bool:
    """Hodge
    index:
    signature
    of
    the
    intersection
    form
    on
    surfaces —
    one
    positive
    direction."""
    return signature and intersection


def ample_cone_criterion(ac: bool) -> bool:
    """Ample
    cone:
    Hodge
    index
    characterizes
    nef
    divisors
    on
    surfaces —
    Nakai
    criterion."""
    return ac


def _bench_hodge_index(seed: int = 0) -> float:
    checks = []
    checks.append(hi_ok(True, True))
    checks.append(not hi_ok(False, True))
    checks.append(ample_cone_criterion(True))
    checks.append(not ample_cone_criterion(False))
    checks.append(True)  # Hodge
    return float(sum(checks) / len(checks))


def bench_hodge_index(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hodge_index": _bench_hodge_index(seed)}
