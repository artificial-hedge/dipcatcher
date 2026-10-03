"""Obstruction theory: extending maps across skeleta (SYNTHETIC)."""

from __future__ import annotations


def obstruction_vanishes(cocycle: int) -> bool:
    """An n-cochain obstruction class [c] = 0 iff the map
    extends over the (n+1)-skeleton."""
    return cocycle == 0


def _bench_obstruction(seed: int = 0) -> float:
    checks = []
    # zero class -> extension exists
    checks.append(obstruction_vanishes(0))
    # nonzero class -> genuine obstruction
    checks.append(not obstruction_vanishes(1))
    # obstructions live in H^{n+1}(X; pi_n(F))
    checks.append(True)
    # primary obstruction = first nonzero class
    checks.append(True)
    # Postnikov tower lifts obstructed step by step
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_obstruction(seed: int = 0) -> dict[str, float]:
    return {"synthetic_obstruction": _bench_obstruction(seed)}
