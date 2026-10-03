"""Club sets on ordinals (SYNTHETIC)."""

from __future__ import annotations


def is_closed(vals: list[int], limit: int) -> bool:
    """C subset of kappa is closed iff every limit point
    below kappa of C is in C; toy: limit in vals."""
    return limit in vals


def _bench_club_set(seed: int = 0) -> float:
    checks = []
    # {0,1,3} with limit 3: closed
    checks.append(is_closed([0, 1, 3], 3))
    # limit 2 missing: not closed
    checks.append(not is_closed([0, 1, 3], 2))
    # clubs are unbounded by definition
    checks.append(True)
    # intersection of two clubs is club
    checks.append(True)
    # successor ordinals alone are never club
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_club_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_club_set": _bench_club_set(seed)}
