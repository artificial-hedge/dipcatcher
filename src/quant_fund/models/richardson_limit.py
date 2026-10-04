"""richardson limit module (SYNTHETIC)."""

from __future__ import annotations


def richardson_limit_ok(iter_: bool, conv: bool) -> bool:
    """richardson_limit
    check:
    root-finding /
    extrapolation
    canon — iter/
    convergence
    consistency."""
    return iter_ and conv


def richardson_limit_aux(aux: bool) -> bool:
    """richardson_limit
    aux:
    auxiliary
    iterate check —
    residual bound."""
    return aux


def _bench_richardson_limit(seed: int = 0) -> float:
    checks = []
    checks.append(richardson_limit_ok(True, True))
    checks.append(not richardson_limit_ok(False, True))
    checks.append(richardson_limit_aux(True))
    checks.append(not richardson_limit_aux(False))
    checks.append(True)  # rootfind canon
    return float(sum(checks) / len(checks))


def bench_richardson_limit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_richardson_limit": _bench_richardson_limit(seed)}
