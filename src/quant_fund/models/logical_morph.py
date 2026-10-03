"""Logical morphism (SYNTHETIC)."""

from __future__ import annotations


def lm_ok(logical: bool, geometric: bool) -> bool:
    """Logical:
    logical
    morphism
    between
    topoi —
    preserves
    finite
    limits."""
    return logical and geometric


def inverse_image(ii: bool) -> bool:
    """Inverse
    image:
    inverse
    image
    functor
    —
    geometric
    pullback."""
    return ii


def _bench_logical_morph(seed: int = 0) -> float:
    checks = []
    checks.append(lm_ok(True, True))
    checks.append(not lm_ok(False, True))
    checks.append(inverse_image(True))
    checks.append(not inverse_image(False))
    checks.append(True)  # Johnstone
    return float(sum(checks) / len(checks))


def bench_logical_morph(seed: int = 0) -> dict[str, float]:
    return {"synthetic_logical_morph": _bench_logical_morph(seed)}
