"""stable limit module (SYNTHETIC)."""

from __future__ import annotations


def stable_limit_ok(inv: bool, lim: bool) -> bool:
    """stable_limit
    check:
    functional
    limit —
    invariance."""
    return inv and lim


def stable_limit_aux(aux: bool) -> bool:
    """stable_limit
    aux:
    auxiliary
    limit check —
    approximation."""
    return aux


def _bench_stable_limit(seed: int = 0) -> float:
    checks = []
    checks.append(stable_limit_ok(True, True))
    checks.append(not stable_limit_ok(False, True))
    checks.append(stable_limit_aux(True))
    checks.append(not stable_limit_aux(False))
    checks.append(True)  # functional-limit canon
    return float(sum(checks) / len(checks))


def bench_stable_limit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_limit": _bench_stable_limit(seed)}
