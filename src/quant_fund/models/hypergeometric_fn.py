"""hypergeometric fn module (SYNTHETIC)."""

from __future__ import annotations


def hypergeometric_fn_ok(arg: bool, order: bool) -> bool:
    """hypergeometric_fn
    check:
    special
    function —
    argument."""
    return arg and order


def hypergeometric_fn_aux(aux: bool) -> bool:
    """hypergeometric_fn
    aux:
    auxiliary
    special check —
    parameter."""
    return aux


def _bench_hypergeometric_fn(seed: int = 0) -> float:
    checks = []
    checks.append(hypergeometric_fn_ok(True, True))
    checks.append(not hypergeometric_fn_ok(False, True))
    checks.append(hypergeometric_fn_aux(True))
    checks.append(not hypergeometric_fn_aux(False))
    checks.append(True)  # special-functions canon
    return float(sum(checks) / len(checks))


def bench_hypergeometric_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hypergeometric_fn": _bench_hypergeometric_fn(seed)}
