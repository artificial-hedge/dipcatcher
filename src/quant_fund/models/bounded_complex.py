"""Bounded derived category: cohomology and truncations (SYNTHETIC)."""

from __future__ import annotations


def truncate_geq(terms: list[int], degree: int) -> list[int]:
    """tau_{>=d} keeps terms in degrees >= d."""
    return [t for i, t in enumerate(terms) if i >= degree]


def cohomology_len(terms: list[int], exact_at: list[int]) -> int:
    """Number of nonzero cohomology groups of a toy complex."""
    return sum(1 for i, t in enumerate(terms) if t != 0 and i not in exact_at)


def _bench_bounded_complex(seed: int = 0) -> float:
    checks = []
    cx = [0, 4, 0, 8, 0]
    checks.append(truncate_geq(cx, 2) == [0, 8, 0])
    checks.append(cohomology_len(cx, [1]) == 1)
    # exact complex: all cohomology zero
    checks.append(cohomology_len([2, 2, 2], [0, 1, 2]) == 0)
    # bounded: truncation of bounded is bounded
    checks.append(len(truncate_geq(cx, 3)) == 2)
    # tau_<=0 then tau_>=1 covers degrees
    checks.append(len(truncate_geq(cx, 0)) == 5)
    return float(sum(checks) / len(checks))


def bench_bounded_complex(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bounded_complex": _bench_bounded_complex(seed)}
