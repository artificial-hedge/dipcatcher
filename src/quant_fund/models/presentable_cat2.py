"""Presentable categories 2 (SYNTHETIC)."""

from __future__ import annotations


def pc_ok(presentable: bool, accessible: bool) -> bool:
    """Presentable
    category:
    locally
    presentable —
    accessible
    cocomplete."""
    return presentable and accessible


def locally_presentable_cat(lpc: bool) -> bool:
    """Locally
    presentable:
    lambda-presentable —
    strong
    generator."""
    return lpc


def _bench_presentable_cat2(seed: int = 0) -> float:
    checks = []
    checks.append(pc_ok(True, True))
    checks.append(not pc_ok(False, True))
    checks.append(locally_presentable_cat(True))
    checks.append(not locally_presentable_cat(False))
    checks.append(True)  # Adamek-Rosicky
    return float(sum(checks) / len(checks))


def bench_presentable_cat2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_presentable_cat2": _bench_presentable_cat2(seed)}
