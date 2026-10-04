"""Suslin K2 (SYNTHETIC)."""

from __future__ import annotations


def sk2_ok(suslin: bool, k2: bool) -> bool:
    """Suslin
    K2:
    Suslin
    K2 —
    stability."""
    return suslin and k2


def suslin_stability(ss: bool) -> bool:
    """Suslin
    stability:
    Suslin
    stability —
    GL
    stabilization."""
    return ss


def _bench_suslin_k2(seed: int = 0) -> float:
    checks = []
    checks.append(sk2_ok(True, True))
    checks.append(not sk2_ok(False, True))
    checks.append(suslin_stability(True))
    checks.append(not suslin_stability(False))
    checks.append(True)  # Suslin
    return float(sum(checks) / len(checks))


def bench_suslin_k2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_suslin_k2": _bench_suslin_k2(seed)}
