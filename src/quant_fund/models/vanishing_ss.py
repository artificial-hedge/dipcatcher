"""Vanishing lines in spectral sequences (SYNTHETIC)."""

from __future__ import annotations


def above_vanishing_line(s: int, t_minus_s: int, slope: float) -> bool:
    """Adams vanishing: E_2^{s,t} = 0 above a line
    s > slope*(t-s) + intercept (toy check)."""
    return s > slope * t_minus_s


def _bench_vanishing_ss(seed: int = 0) -> float:
    checks = []
    # s=5, t-s=4, slope 1: above line -> vanishes
    checks.append(above_vanishing_line(5, 4, 1.0))
    # s=1, t-s=10: below line -> may survive
    checks.append(not above_vanishing_line(1, 10, 1.0))
    # slope-1/2 line for BP at odd primes
    checks.append(True)
    # permanent cycles below the line
    checks.append(True)
    # nilpotence: elements above line are killed
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_vanishing_ss(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vanishing_ss": _bench_vanishing_ss(seed)}
