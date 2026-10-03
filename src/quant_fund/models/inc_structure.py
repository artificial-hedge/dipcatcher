"""Incidence structure of PG(2, q) (SYNTHETIC)."""

from __future__ import annotations


def pg2_points(q: int) -> int:
    """Number of points (= lines) of PG(2,q): q^2 + q + 1."""
    return q * q + q + 1


def pg2_line_size(q: int) -> int:
    """Points per line of PG(2,q): q + 1."""
    return q + 1


def pg2_flag_count(q: int) -> int:
    """Incidence flags: (q^2+q+1)(q+1)."""
    return pg2_points(q) * pg2_line_size(q)


def _bench_inc_structure(seed: int = 0) -> float:
    checks = []
    # Fano: PG(2,2) has 7 points, 3 per line
    checks.append(pg2_points(2) == 7)
    checks.append(pg2_line_size(2) == 3)
    # PG(2,3): 13 points, 4 per line
    checks.append(pg2_points(3) == 13)
    checks.append(pg2_line_size(3) == 4)
    # PG(2,4): 21 points
    checks.append(pg2_points(4) == 21)
    # flag count of Fano = 21
    checks.append(pg2_flag_count(2) == 21)
    # double counting: flags via points = via lines (same formula)
    checks.append(pg2_flag_count(3) == 13 * 4)
    return float(sum(checks) / len(checks))


def bench_inc_structure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inc_structure": _bench_inc_structure(seed)}
