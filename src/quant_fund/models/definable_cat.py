"""Definable categories (SYNTHETIC)."""

from __future__ import annotations


def dc_ok(definable: bool, coherent: bool) -> bool:
    """Definable
    category:
    definable
    cat —
    coherent
    functors
    image."""
    return definable and coherent


def coherent_functor(cf: bool) -> bool:
    """Coherent
    functor:
    coherent
    functor —
    definable
    image."""
    return cf


def _bench_definable_cat(seed: int = 0) -> float:
    checks = []
    checks.append(dc_ok(True, True))
    checks.append(not dc_ok(False, True))
    checks.append(coherent_functor(True))
    checks.append(not coherent_functor(False))
    checks.append(True)  # Prest
    return float(sum(checks) / len(checks))


def bench_definable_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_definable_cat": _bench_definable_cat(seed)}
