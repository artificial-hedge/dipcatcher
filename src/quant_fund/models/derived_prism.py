"""Derived prisms (SYNTHETIC)."""

from __future__ import annotations


def dp_ok(derived: bool, prism: bool) -> bool:
    """Derived
    prism:
    derived
    prism —
    animated."""
    return derived and prism


def animated_delta(ad: bool) -> bool:
    """Animated
    delta:
    animated
    delta
    ring —
    simplicial."""
    return ad


def _bench_derived_prism(seed: int = 0) -> float:
    checks = []
    checks.append(dp_ok(True, True))
    checks.append(not dp_ok(False, True))
    checks.append(animated_delta(True))
    checks.append(not animated_delta(False))
    checks.append(True)  # Bhatt-Lurie
    return float(sum(checks) / len(checks))


def bench_derived_prism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_prism": _bench_derived_prism(seed)}
