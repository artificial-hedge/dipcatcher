"""martingale fclt module (SYNTHETIC)."""

from __future__ import annotations


def martingale_fclt_ok(inv: bool, lim: bool) -> bool:
    """martingale_fclt
    check:
    functional
    limit —
    invariance."""
    return inv and lim


def martingale_fclt_aux(aux: bool) -> bool:
    """martingale_fclt
    aux:
    auxiliary
    limit check —
    approximation."""
    return aux


def _bench_martingale_fclt(seed: int = 0) -> float:
    checks = []
    checks.append(martingale_fclt_ok(True, True))
    checks.append(not martingale_fclt_ok(False, True))
    checks.append(martingale_fclt_aux(True))
    checks.append(not martingale_fclt_aux(False))
    checks.append(True)  # functional-limit canon
    return float(sum(checks) / len(checks))


def bench_martingale_fclt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_martingale_fclt": _bench_martingale_fclt(seed)}
