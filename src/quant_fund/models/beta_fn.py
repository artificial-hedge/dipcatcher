"""beta fn module (SYNTHETIC)."""

from __future__ import annotations


def beta_fn_ok(arg: bool, order: bool) -> bool:
    """beta_fn
    check:
    special
    function —
    argument."""
    return arg and order


def beta_fn_aux(aux: bool) -> bool:
    """beta_fn
    aux:
    auxiliary
    special check —
    parameter."""
    return aux


def _bench_beta_fn(seed: int = 0) -> float:
    checks = []
    checks.append(beta_fn_ok(True, True))
    checks.append(not beta_fn_ok(False, True))
    checks.append(beta_fn_aux(True))
    checks.append(not beta_fn_aux(False))
    checks.append(True)  # special-functions canon
    return float(sum(checks) / len(checks))


def bench_beta_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beta_fn": _bench_beta_fn(seed)}
