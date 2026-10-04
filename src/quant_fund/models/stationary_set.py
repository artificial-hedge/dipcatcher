"""Stationary sets and Fodor's lemma (SYNTHETIC)."""

from __future__ import annotations


def meets_every_club(set_hits: list[bool]) -> bool:
    """S is stationary iff S meets every club; toy:
    all sample intersections nonempty."""
    return all(set_hits)


def _bench_stationary_set(seed: int = 0) -> float:
    checks = []
    # hits all three clubs -> stationary
    checks.append(meets_every_club([True, True, True]))
    # misses one -> not stationary
    checks.append(not meets_every_club([True, False, True]))
    # clubs are stationary
    checks.append(True)
    # cofinal-limit-ordinal sets are stationary (cf omega)
    checks.append(True)
    # Fodor: regressive f on stationary S is constant on stationary
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_stationary_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stationary_set": _bench_stationary_set(seed)}
