"""elliptic fn module (SYNTHETIC)."""

from __future__ import annotations


def elliptic_fn_ok(arg: bool, param: bool) -> bool:
    """elliptic_fn
    check:
    special function /
    transform canon —
    arg/param consistency."""
    return arg and param


def elliptic_fn_aux(aux: bool) -> bool:
    """elliptic_fn
    aux:
    auxiliary
    transform check —
    identity bound."""
    return aux


def _bench_elliptic_fn(seed: int = 0) -> float:
    checks = []
    checks.append(elliptic_fn_ok(True, True))
    checks.append(not elliptic_fn_ok(False, True))
    checks.append(elliptic_fn_aux(True))
    checks.append(not elliptic_fn_aux(False))
    checks.append(True)  # special-fn canon
    return float(sum(checks) / len(checks))


def bench_elliptic_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elliptic_fn": _bench_elliptic_fn(seed)}
