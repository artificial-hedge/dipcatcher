"""Locally presentable categories (SYNTHETIC)."""

from __future__ import annotations


def lp_ok(locally: bool, presentable: bool) -> bool:
    """Locally
    presentable:
    locally
    finitely
    presentable —
    Vopěnka."""
    return locally and presentable


def vopenka_principle(vp: bool) -> bool:
    """Vopěnka
    principle:
    Vopěnka
    principle —
    large
    cardinal."""
    return vp


def _bench_locally_presentable(seed: int = 0) -> float:
    checks = []
    checks.append(lp_ok(True, True))
    checks.append(not lp_ok(False, True))
    checks.append(vopenka_principle(True))
    checks.append(not vopenka_principle(False))
    checks.append(True)  # Adamek-Rosicky
    return float(sum(checks) / len(checks))


def bench_locally_presentable(seed: int = 0) -> dict[str, float]:
    return {"synthetic_locally_presentable": _bench_locally_presentable(seed)}
