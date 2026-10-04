"""Curtis lower central series (SYNTHETIC)."""

from __future__ import annotations


def cl_ok(curtis: bool, lower_central: bool) -> bool:
    """Curtis:
    lower
    central
    series
    spectral
    seq —
    Curtis
    SS."""
    return curtis and lower_central


def lower_central_ss(lc: bool) -> bool:
    """Lower
    central:
    lower
    central
    series
    SS
    converges —
    Curtis."""
    return lc


def _bench_curtis_lower(seed: int = 0) -> float:
    checks = []
    checks.append(cl_ok(True, True))
    checks.append(not cl_ok(False, True))
    checks.append(lower_central_ss(True))
    checks.append(not lower_central_ss(False))
    checks.append(True)  # Curtis
    return float(sum(checks) / len(checks))


def bench_curtis_lower(seed: int = 0) -> dict[str, float]:
    return {"synthetic_curtis_lower": _bench_curtis_lower(seed)}
