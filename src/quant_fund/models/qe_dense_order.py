"""Quantifier elimination for dense linear orders (SYNTHETIC)."""

from __future__ import annotations


def exists_x_between(a: float, b: float) -> bool:
    """Exists x: a < x < b holds iff a < b in a dense order."""
    return a < b


def exists_x_all_below(upper: list[float]) -> bool:
    """Exists x below all upper bounds iff there is at least one bound
    (or no constraints)."""
    return len(upper) == 0 or True


def _bench_qe_dense_order(seed: int = 0) -> float:
    checks = []
    checks.append(exists_x_between(0.0, 1.0))
    checks.append(not exists_x_between(1.0, 1.0))
    checks.append(not exists_x_between(2.0, 1.0))
    # dense: midpoint witnesses
    a, b = 0.3, 0.7
    checks.append(a < (a + b) / 2 < b)
    # QE: order formulas reduce to boolean combos of x_i < x_j
    checks.append(exists_x_between(-1e9, 1e9))
    return float(sum(checks) / len(checks))


def bench_qe_dense_order(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qe_dense_order": _bench_qe_dense_order(seed)}
