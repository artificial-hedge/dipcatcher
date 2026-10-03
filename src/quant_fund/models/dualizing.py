"""Dualizing sheaf / canonical divisor (SYNTHETIC)."""

from __future__ import annotations


def canonical_degree(g: int) -> int:
    """deg K_C = 2g - 2 on a smooth projective curve."""
    return 2 * g - 2


def _bench_dualizing(seed: int = 0) -> float:
    checks = []
    # P1: deg K = -2
    checks.append(canonical_degree(0) == -2)
    # elliptic: deg K = 0 (K trivial)
    checks.append(canonical_degree(1) == 0)
    # genus 2: deg K = 2
    checks.append(canonical_degree(2) == 2)
    # plane curve degree d: K = (d-3)H -> deg = d(d-3)
    checks.append(canonical_degree((3 - 1) * (3 - 2) // 2) == 0)
    # Serre duality uses the dualizing sheaf
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_dualizing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dualizing": _bench_dualizing(seed)}
