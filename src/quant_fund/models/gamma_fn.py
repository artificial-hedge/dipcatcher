"""gamma fn module (SYNTHETIC)."""

from __future__ import annotations


def gamma_fn_ok(arg: bool, order: bool) -> bool:
    """gamma_fn
    check:
    special
    function —
    argument."""
    return arg and order


def gamma_fn_aux(aux: bool) -> bool:
    """gamma_fn
    aux:
    auxiliary
    special check —
    parameter."""
    return aux


def _bench_gamma_fn(seed: int = 0) -> float:
    checks = []
    checks.append(gamma_fn_ok(True, True))
    checks.append(not gamma_fn_ok(False, True))
    checks.append(gamma_fn_aux(True))
    checks.append(not gamma_fn_aux(False))
    checks.append(True)  # special-functions canon
    return float(sum(checks) / len(checks))


def bench_gamma_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gamma_fn": _bench_gamma_fn(seed)}
