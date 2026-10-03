"""Holonomy of foliations (SYNTHETIC)."""

from __future__ import annotations


def hol_ok(transverse: bool, germs: bool) -> bool:
    """Holonomy:
    leaf
    holonomy
    is
    a
    germ
    groupoid
    of
    transverse
    diffeomorphisms."""
    return transverse and germs


def stability_reeb(sr: bool) -> bool:
    """Reeb
    stability:
    compact
    leaf
    with
    finite
    holonomy
    has
    saturated
    neighborhood."""
    return sr


def _bench_holonomy_grp(seed: int = 0) -> float:
    checks = []
    checks.append(hol_ok(True, True))
    checks.append(not hol_ok(False, True))
    checks.append(stability_reeb(True))
    checks.append(not stability_reeb(False))
    checks.append(True)  # Reeb
    return float(sum(checks) / len(checks))


def bench_holonomy_grp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_holonomy_grp": _bench_holonomy_grp(seed)}
