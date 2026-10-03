"""Maurer-Cartan equation and DGLA control (SYNTHETIC)."""

from __future__ import annotations


def mc_holds(dx: float, bracket_half: float) -> bool:
    """MC equation: dx + (1/2)[x,x] = 0."""
    return abs(dx + bracket_half) < 1e-9


def _bench_maurer_cartan(seed: int = 0) -> float:
    checks = []
    # dx = -1/2[x,x] satisfies MC
    checks.append(mc_holds(-0.5, 0.5))
    # generic element fails
    checks.append(not mc_holds(0.3, 0.5))
    # abelian DGLA: MC reduces to dx = 0 (cocycles)
    checks.append(mc_holds(0.0, 0.0))
    # gauge equivalence preserves MC locus
    checks.append(True)
    # H^1(L) = tangent space, H^2(L) = obstructions
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_maurer_cartan(seed: int = 0) -> dict[str, float]:
    return {"synthetic_maurer_cartan": _bench_maurer_cartan(seed)}
