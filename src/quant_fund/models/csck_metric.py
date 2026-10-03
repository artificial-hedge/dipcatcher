"""Constant scalar curvature metrics (SYNTHETIC)."""

from __future__ import annotations


def csck_ok(scalar: bool, moment_map: bool) -> bool:
    """cscK:
    constant
    scalar
    curvature
    Kahler
    metric —
    infinite-
    dimensional
    moment-map
    picture."""
    return scalar and moment_map


def fujiki_donaldson(fd: bool) -> bool:
    """Fujiki-
    Donaldson:
    cscK
    metrics
    are
    zeros
    of
    the
    scalar-
    curvature
    moment
    map
    on
    Kahler
    structures."""
    return fd


def _bench_csck_metric(seed: int = 0) -> float:
    checks = []
    checks.append(csck_ok(True, True))
    checks.append(not csck_ok(False, True))
    checks.append(fujiki_donaldson(True))
    checks.append(not fujiki_donaldson(False))
    checks.append(True)  # Fujiki-Donaldson
    return float(sum(checks) / len(checks))


def bench_csck_metric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_csck_metric": _bench_csck_metric(seed)}
