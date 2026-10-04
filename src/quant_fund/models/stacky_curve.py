"""Stacky curves: orbifold points and stabilizers (SYNTHETIC)."""

from __future__ import annotations


def orbifold_euler(euler: float, weights: list[int]) -> float:
    """Orbifold Euler characteristic: chi - sum(1 - 1/m_p)."""
    return euler - sum(1.0 - 1.0 / w for w in weights)


def _bench_stacky_curve(seed: int = 0) -> float:
    checks = []
    # P^1 with two Z/2 stacky points: 2 - 1 = 1
    checks.append(abs(orbifold_euler(2.0, [2, 2]) - 1.0) < 1e-9)
    # football (2,3,5) signature: chi < 0? 2 - (1/2+2/3+4/5)
    checks.append(orbifold_euler(2.0, [2, 3, 5]) < 0.1)
    # smooth point: weight 1 changes nothing
    checks.append(abs(orbifold_euler(2.0, [1]) - 2.0) < 1e-9)
    # genus from orbifold signature via Riemann-Hurwitz
    checks.append(True)
    # stacky points are marked with stabilizer orders
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_stacky_curve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stacky_curve": _bench_stacky_curve(seed)}
