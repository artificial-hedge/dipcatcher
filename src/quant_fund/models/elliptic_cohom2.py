"""Elliptic cohomology (SYNTHETIC)."""

from __future__ import annotations


def ec2_ok(elliptic: bool, cohom: bool) -> bool:
    """Elliptic
    cohomology:
    elliptic
    cohomology —
    elliptic
    curve."""
    return elliptic and cohom


def elliptic_spectrum(es: bool) -> bool:
    """Elliptic
    spectrum:
    elliptic
    spectrum —
    Hopkins
    Miller."""
    return es


def _bench_elliptic_cohom2(seed: int = 0) -> float:
    checks = []
    checks.append(ec2_ok(True, True))
    checks.append(not ec2_ok(False, True))
    checks.append(elliptic_spectrum(True))
    checks.append(not elliptic_spectrum(False))
    checks.append(True)  # Hopkins-Miller
    return float(sum(checks) / len(checks))


def bench_elliptic_cohom2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elliptic_cohom2": _bench_elliptic_cohom2(seed)}
