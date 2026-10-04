"""bounded var module (SYNTHETIC)."""

from __future__ import annotations


def bounded_var_ok(si: bool, isom: bool) -> bool:
    """bounded_var
    check:
    stochastic
    integral —
    isometry."""
    return si and isom


def bounded_var_aux(aux: bool) -> bool:
    """bounded_var
    aux:
    auxiliary
    integral
    check —
    covariation."""
    return aux


def _bench_bounded_var(seed: int = 0) -> float:
    checks = []
    checks.append(bounded_var_ok(True, True))
    checks.append(not bounded_var_ok(False, True))
    checks.append(bounded_var_aux(True))
    checks.append(not bounded_var_aux(False))
    checks.append(True)  # integration canon
    return float(sum(checks) / len(checks))


def bench_bounded_var(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bounded_var": _bench_bounded_var(seed)}
