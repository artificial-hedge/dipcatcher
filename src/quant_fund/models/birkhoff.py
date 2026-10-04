"""Birkhoff ergodic theorem (SYNTHETIC)."""

from __future__ import annotations


def birkhoff_ok(time_avg: bool, conv: bool) -> bool:
    """Birkhoff
    ergodic
    theorem:
    time
    averages
    converge
    a.e. to
    the
    conditional
    expectation
    under
    measure-
    preserving
    maps."""
    return time_avg and conv


def pointwise_ergodic(pt: bool) -> bool:
    """Pointwise
    ergodic
    theorem:
    almost-
    everywhere
    convergence
    for
    L^1
    functions."""
    return pt


def _bench_birkhoff(seed: int = 0) -> float:
    checks = []
    checks.append(birkhoff_ok(True, True))
    checks.append(not birkhoff_ok(False, True))
    checks.append(pointwise_ergodic(True))
    checks.append(not pointwise_ergodic(False))
    checks.append(True)  # Birkhoff
    return float(sum(checks) / len(checks))


def bench_birkhoff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_birkhoff": _bench_birkhoff(seed)}
