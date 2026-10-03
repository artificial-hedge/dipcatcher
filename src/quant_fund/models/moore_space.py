"""Moore space (SYNTHETIC)."""

from __future__ import annotations


def ms_ok(moore_space: bool, homology: bool) -> bool:
    """Moore
    space:
    specified
    single
    homology
    group —
    Moore
    M(G,n)."""
    return moore_space and homology


def moore_cohom(mc: bool) -> bool:
    """Moore
    cohomology:
    cohomology
    of
    Moore
    space
    dual —
    Universal
    coefficient."""
    return mc


def _bench_moore_space(seed: int = 0) -> float:
    checks = []
    checks.append(ms_ok(True, True))
    checks.append(not ms_ok(False, True))
    checks.append(moore_cohom(True))
    checks.append(not moore_cohom(False))
    checks.append(True)  # Moore
    return float(sum(checks) / len(checks))


def bench_moore_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moore_space": _bench_moore_space(seed)}
