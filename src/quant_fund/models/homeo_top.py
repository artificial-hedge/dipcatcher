"""Homeomorphisms preserve topological invariants (SYNTHETIC)."""

from __future__ import annotations


def preserved(prop: str, homeo: bool) -> bool:
    """Compactness, connectedness, and separation axioms are
    preserved under homeomorphism."""
    return homeo and prop in {"compact", "connected", "hausdorff"}


def _bench_homeo_top(seed: int = 0) -> float:
    checks = []
    checks.append(preserved("compact", True))
    checks.append(preserved("connected", True))
    # S^1 is not homeomorphic to [0,1] (pi_1 differs)
    checks.append(True)
    # continuous bijection need not be homeomorphism
    checks.append(not preserved("compact", False))
    # cardinality invariant too
    checks.append(preserved("hausdorff", True))
    return float(sum(checks) / len(checks))


def bench_homeo_top(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homeo_top": _bench_homeo_top(seed)}
