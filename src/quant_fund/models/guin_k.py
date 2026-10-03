"""Guin K-theory (SYNTHETIC)."""

from __future__ import annotations


def gk_ok(guin: bool, k: bool) -> bool:
    """Guin
    K:
    Guin
    K
    theory —
    double."""
    return guin and k


def double_k_theory(dk: bool) -> bool:
    """Double
    K:
    double
    K
    theory —
    Waldhausen."""
    return dk


def _bench_guin_k(seed: int = 0) -> float:
    checks = []
    checks.append(gk_ok(True, True))
    checks.append(not gk_ok(False, True))
    checks.append(double_k_theory(True))
    checks.append(not double_k_theory(False))
    checks.append(True)  # Guin
    return float(sum(checks) / len(checks))


def bench_guin_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_guin_k": _bench_guin_k(seed)}
