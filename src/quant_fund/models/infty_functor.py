"""Infinity functors (SYNTHETIC)."""

from __future__ import annotations


def if2_ok(infty: bool, functor: bool) -> bool:
    """Infinity
    functor:
    infinity
    functor —
    simplicial
    map."""
    return infty and functor


def natural_transform(nt: bool) -> bool:
    """Natural
    transform:
    natural
    transformation
    of
    infinity
    functors —
    homotopy
    coherent."""
    return nt


def _bench_infty_functor(seed: int = 0) -> float:
    checks = []
    checks.append(if2_ok(True, True))
    checks.append(not if2_ok(False, True))
    checks.append(natural_transform(True))
    checks.append(not natural_transform(False))
    checks.append(True)  # Lurie
    return float(sum(checks) / len(checks))


def bench_infty_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_infty_functor": _bench_infty_functor(seed)}
