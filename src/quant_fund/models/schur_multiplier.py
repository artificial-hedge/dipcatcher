"""Schur multiplier M(G) = H_2(G, Z) toy table (SYNTHETIC)."""

from __future__ import annotations

_MULT = {
    "C1": 1,
    "C2": 1,
    "C3": 1,
    "C4": 1,
    "V4": 2,
    "S3": 1,
    "A4": 2,
    "Q8": 1,
    "D4": 2,
    "A5": 2,
}


def multiplier_order(group: str) -> int:
    """Order of the Schur multiplier for the toy table."""
    return _MULT[group]


def covers_from_mult(m: int) -> int:
    """Number of covering directions to check: |M| many representation
    classes for the stem covers (model: m distinct central extensions)."""
    return m


def _bench_schur_multiplier(seed: int = 0) -> float:
    checks = []
    # cyclic groups have trivial multiplier
    checks.append(multiplier_order("C2") == 1)
    checks.append(multiplier_order("C4") == 1)
    # V4 has M = C2: V4 = Q8/Z(Q8), the stem extension doubles
    checks.append(multiplier_order("V4") == 2)
    checks.append(multiplier_order("D4") == 2)
    # A4, A5 have multiplier C2 (binary covers 2.A4, 2.A5 = SL(2,5))
    checks.append(multiplier_order("A4") == 2)
    checks.append(multiplier_order("A5") == 2)
    # S3 has trivial multiplier
    checks.append(multiplier_order("S3") == 1)
    # covers_from_mult tracks table size
    checks.append(covers_from_mult(2) == 2)
    return float(sum(checks) / len(checks))


def bench_schur_multiplier(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schur_multiplier": _bench_schur_multiplier(seed)}
