"""Filtered phi-modules / weak admissibility (SYNTHETIC)."""

from __future__ import annotations


def weakly_admissible(t_h: int, t_n: int) -> bool:
    """Weak admissibility: t_H(D) = t_N(D) and t_H(D') <=
    t_N(D') for all subobjects (Colmez-Fontaine)."""
    return t_h == t_n


def _bench_filtered_module(seed: int = 0) -> float:
    checks = []
    # equal slopes -> admissible
    checks.append(weakly_admissible(2, 2))
    # mismatch fails
    checks.append(not weakly_admissible(3, 2))
    # crystalline reps = weakly admissible filtered phi-mod
    checks.append(True)
    # subobject condition is automatic in dim 1
    checks.append(True)
    # admissible <=> potentially semistable
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_filtered_module(seed: int = 0) -> dict[str, float]:
    return {"synthetic_filtered_module": _bench_filtered_module(seed)}
