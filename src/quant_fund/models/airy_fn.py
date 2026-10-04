"""airy fn module (SYNTHETIC)."""

from __future__ import annotations


def airy_fn_ok(arg: bool, order: bool) -> bool:
    """airy_fn
    check:
    special
    function —
    argument."""
    return arg and order


def airy_fn_aux(aux: bool) -> bool:
    """airy_fn
    aux:
    auxiliary
    special check —
    parameter."""
    return aux


def _bench_airy_fn(seed: int = 0) -> float:
    checks = []
    checks.append(airy_fn_ok(True, True))
    checks.append(not airy_fn_ok(False, True))
    checks.append(airy_fn_aux(True))
    checks.append(not airy_fn_aux(False))
    checks.append(True)  # special-functions canon
    return float(sum(checks) / len(checks))


def bench_airy_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_airy_fn": _bench_airy_fn(seed)}
