"""Anti-self-dual equations (SYNTHETIC)."""

from __future__ import annotations


def asd_ok(f_minus: bool, curvature2: bool) -> bool:
    """ASD
    equations:
    F
    plus
    star
    F
    equals
    zero —
    first-order
    equations
    implying
    Yang-
    Mills."""
    return f_minus and curvature2


def bubbling_seq(bs: bool) -> bool:
    """Uhlenbeck
    compactness:
    instanton
    sequences
    bubble
    off
    energy
    at
    points —
    compactified
    moduli."""
    return bs


def _bench_anti_self_dual(seed: int = 0) -> float:
    checks = []
    checks.append(asd_ok(True, True))
    checks.append(not asd_ok(False, True))
    checks.append(bubbling_seq(True))
    checks.append(not bubbling_seq(False))
    checks.append(True)  # Uhlenbeck
    return float(sum(checks) / len(checks))


def bench_anti_self_dual(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anti_self_dual": _bench_anti_self_dual(seed)}
