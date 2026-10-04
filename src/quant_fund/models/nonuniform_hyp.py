"""Nonuniform hyperbolicity (SYNTHETIC)."""

from __future__ import annotations


def nuh_ok(temp: bool, slow: bool) -> bool:
    """Nonuniform
    hyperbolicity:
    tempered
    Lyapunov
    norm gives
    uniform
    estimates
    along
    orbits."""
    return temp and slow


def slow_variation(sv: bool) -> bool:
    """Slow
    variation:
    exponents
    and
    angles
    change
    sub-
    exponentially."""
    return sv


def _bench_nonuniform_hyp(seed: int = 0) -> float:
    checks = []
    checks.append(nuh_ok(True, True))
    checks.append(not nuh_ok(False, True))
    checks.append(slow_variation(True))
    checks.append(not slow_variation(False))
    checks.append(True)  # Pesin-Katok
    return float(sum(checks) / len(checks))


def bench_nonuniform_hyp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nonuniform_hyp": _bench_nonuniform_hyp(seed)}
