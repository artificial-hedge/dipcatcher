"""Semiuniversal deformation (SYNTHETIC)."""

from __future__ import annotations


def su_ok(semiuniversal: bool, hull: bool) -> bool:
    """Semiuniversal:
    semiuniversal
    deformation
    hull —
    semiuniversal
    deformation."""
    return semiuniversal and hull


def semiuniv_hull(sh: bool) -> bool:
    """Semiuniversal
    hull:
    semiuniversal
    hull
    construction —
    semiuniversal
    hull."""
    return sh


def _bench_semiuniversal(seed: int = 0) -> float:
    checks = []
    checks.append(su_ok(True, True))
    checks.append(not su_ok(False, True))
    checks.append(semiuniv_hull(True))
    checks.append(not semiuniv_hull(False))
    checks.append(True)  # semiuniversal
    return float(sum(checks) / len(checks))


def bench_semiuniversal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_semiuniversal": _bench_semiuniversal(seed)}
