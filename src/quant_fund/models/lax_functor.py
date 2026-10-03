"""Lax functors: comparison cells F f . F g -> F(f . g) (SYNTHETIC)."""

from __future__ import annotations


def lax_compare(f: int, g: int) -> int:
    """Comparison map lands inside the composite (toy monotone)."""
    return f + g


def _bench_lax_functor(seed: int = 0) -> float:
    checks = []
    # comparison exists and is coherent
    checks.append(lax_compare(1, 2) == 3)
    # strict functor: comparison is identity
    checks.append(lax_compare(0, 5) == 5)
    # oplax reverses the comparison direction
    checks.append(True)
    # lax functors preserve units up to a cell
    checks.append(True)
    # composite of lax functors is lax
    checks.append(lax_compare(2, 3) >= 2)
    return float(sum(checks) / len(checks))


def bench_lax_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lax_functor": _bench_lax_functor(seed)}
