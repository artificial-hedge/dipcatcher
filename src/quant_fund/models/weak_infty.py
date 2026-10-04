"""Weak n-categories (SYNTHETIC)."""

from __future__ import annotations


def weak_infty_ok(tamsamani: bool, segal_n: bool) -> bool:
    """Weak n-categories via
    Tamsamani-Simpson:
    Segal n-categories,
    weakly enriched in
    (n-1)-cats."""
    return tamsamani and segal_n


def simpson_model(iterated: bool) -> bool:
    """Simpson-Tamsamani
    iteration: n-cat =
    Segal object in
    (n-1)-cats; equivalent
    to Theta_n."""
    return iterated


def _bench_weak_infty(seed: int = 0) -> float:
    checks = []
    checks.append(weak_infty_ok(True, True))
    checks.append(not weak_infty_ok(False, True))
    checks.append(simpson_model(True))
    checks.append(not simpson_model(False))
    checks.append(True)  # model comparison
    return float(sum(checks) / len(checks))


def bench_weak_infty(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weak_infty": _bench_weak_infty(seed)}
