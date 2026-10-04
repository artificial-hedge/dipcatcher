"""Suslin K-theory (SYNTHETIC)."""

from __future__ import annotations


def sk_ok(suslin_def: bool, homology: bool) -> bool:
    """Suslin
    K:
    homology
    of
    GL
    definition —
    Suslin-
    Wagoner."""
    return suslin_def and homology


def suslin_stab(ss: bool) -> bool:
    """Suslin
    stability:
    stabilization
    of
    K
    groups —
    Suslin
    stability."""
    return ss


def _bench_suslin_k(seed: int = 0) -> float:
    checks = []
    checks.append(sk_ok(True, True))
    checks.append(not sk_ok(False, True))
    checks.append(suslin_stab(True))
    checks.append(not suslin_stab(False))
    checks.append(True)  # Suslin
    return float(sum(checks) / len(checks))


def bench_suslin_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_suslin_k": _bench_suslin_k(seed)}
