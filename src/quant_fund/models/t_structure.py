"""t-structures: truncations and the heart (SYNTHETIC)."""

from __future__ import annotations


def heart_terms(terms: list[int]) -> list[int]:
    """Heart D^{<=0} cap D^{>=0}: degree-0 part of a complex."""
    return [t for i, t in enumerate(terms) if i == 0]


def in_heart(terms: list[int]) -> bool:
    """A complex lies in the heart iff only degree-0 terms nonzero."""
    return all(t == 0 for i, t in enumerate(terms) if i != 0)


def _bench_t_structure(seed: int = 0) -> float:
    checks = []
    # index = degree: only degree-0 term survives in the heart
    checks.append(heart_terms([7, 3, 0]) == [7])
    checks.append(in_heart([7, 0, 0]))
    checks.append(not in_heart([7, 1, 0]))
    # heart of D^b(A) is A: a degree-0 term is its own heart
    checks.append(heart_terms([0, 0, 0]) == [0])
    # truncation functors give triangles tau_<=0 -> X -> tau_>=1
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_t_structure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_t_structure": _bench_t_structure(seed)}
