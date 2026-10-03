"""error fn module (SYNTHETIC)."""

from __future__ import annotations


def error_fn_ok(arg: bool, order: bool) -> bool:
    """error_fn
    check:
    special
    function —
    argument."""
    return arg and order


def error_fn_aux(aux: bool) -> bool:
    """error_fn
    aux:
    auxiliary
    special check —
    parameter."""
    return aux


def _bench_error_fn(seed: int = 0) -> float:
    checks = []
    checks.append(error_fn_ok(True, True))
    checks.append(not error_fn_ok(False, True))
    checks.append(error_fn_aux(True))
    checks.append(not error_fn_aux(False))
    checks.append(True)  # special-functions canon
    return float(sum(checks) / len(checks))


def bench_error_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_error_fn": _bench_error_fn(seed)}
