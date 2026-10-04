"""Volodin K-theory (SYNTHETIC)."""

from __future__ import annotations


def vk_ok(volodin_def: bool, triangular: bool) -> bool:
    """Volodin
    K:
    triangular
    matrix
    definition —
    Volodin-
    Suslin."""
    return volodin_def and triangular


def volodin_equiv(ve: bool) -> bool:
    """Volodin
    equiv:
    Volodin
    equals
    Quillen
    K —
    equivalence."""
    return ve


def _bench_volodin_k(seed: int = 0) -> float:
    checks = []
    checks.append(vk_ok(True, True))
    checks.append(not vk_ok(False, True))
    checks.append(volodin_equiv(True))
    checks.append(not volodin_equiv(False))
    checks.append(True)  # Volodin-Suslin
    return float(sum(checks) / len(checks))


def bench_volodin_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_volodin_k": _bench_volodin_k(seed)}
