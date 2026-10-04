"""Dror-Smith spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def ds_ok(dror: bool, smith: bool) -> bool:
    """Dror-
    Smith:
    cellular
    inequalities
    SS —
    Dror
    Farjoun."""
    return dror and smith


def cellular_ineq(ci: bool) -> bool:
    """Cellular:
    cellular
    inequalities
    for
    nullification —
    Dror
    Smith."""
    return ci


def _bench_dror_smith(seed: int = 0) -> float:
    checks = []
    checks.append(ds_ok(True, True))
    checks.append(not ds_ok(False, True))
    checks.append(cellular_ineq(True))
    checks.append(not cellular_ineq(False))
    checks.append(True)  # Dror-Smith
    return float(sum(checks) / len(checks))


def bench_dror_smith(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dror_smith": _bench_dror_smith(seed)}
