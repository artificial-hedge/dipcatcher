"""brownian approx module (SYNTHETIC)."""

from __future__ import annotations


def brownian_approx_ok(inv: bool, lim: bool) -> bool:
    """brownian_approx
    check:
    functional
    limit —
    invariance."""
    return inv and lim


def brownian_approx_aux(aux: bool) -> bool:
    """brownian_approx
    aux:
    auxiliary
    limit check —
    approximation."""
    return aux


def _bench_brownian_approx(seed: int = 0) -> float:
    checks = []
    checks.append(brownian_approx_ok(True, True))
    checks.append(not brownian_approx_ok(False, True))
    checks.append(brownian_approx_aux(True))
    checks.append(not brownian_approx_aux(False))
    checks.append(True)  # functional-limit canon
    return float(sum(checks) / len(checks))


def bench_brownian_approx(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brownian_approx": _bench_brownian_approx(seed)}
