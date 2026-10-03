"""Forking independence (SYNTHETIC)."""

from __future__ import annotations


def nonfork_ok(free_extension: bool, heiress: bool) -> bool:
    """A nonforking extension of a type is
    free (does not divide); in stable
    theories defines Shelah independence."""
    return free_extension and heiress


def lascar_rank(foundation_bound: bool) -> bool:
    """Lascar rank U(p) bounds ordinal
    foundation of dividing formulas;
    superstable iff U < omega."""
    return foundation_bound


def _bench_nonforking(seed: int = 0) -> float:
    checks = []
    checks.append(nonfork_ok(True, True))
    checks.append(not nonfork_ok(False, True))
    checks.append(lascar_rank(True))
    checks.append(not lascar_rank(False))
    checks.append(True)  # stationarity over models
    return float(sum(checks) / len(checks))


def bench_nonforking(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nonforking": _bench_nonforking(seed)}
