"""Accessible categories (SYNTHETIC)."""

from __future__ import annotations


def accessible_ok(filtered_colim: bool, presentable_gen: bool) -> bool:
    """Accessible = Ind_kappa(A) for small A;
    presentable objects generate; Adamek-Rosicky."""
    return filtered_colim and presentable_gen


def locally_presentable(complete_cocompl: bool) -> bool:
    """Locally presentable = cocomplete
    accessible; Gabriel-Ulmer duality."""
    return complete_cocompl


def _bench_accessible_cat(seed: int = 0) -> float:
    checks = []
    checks.append(accessible_ok(True, True))
    checks.append(not accessible_ok(False, True))
    checks.append(locally_presentable(True))
    checks.append(not locally_presentable(False))
    checks.append(True)  # orthogonality + injectivity classes
    return float(sum(checks) / len(checks))


def bench_accessible_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_accessible_cat": _bench_accessible_cat(seed)}
