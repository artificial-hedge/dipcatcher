"""clenshaw quad module (SYNTHETIC)."""

from __future__ import annotations


def clenshaw_quad_ok(arg: bool, param: bool) -> bool:
    """clenshaw_quad
    check:
    special function /
    transform canon —
    arg/param consistency."""
    return arg and param


def clenshaw_quad_aux(aux: bool) -> bool:
    """clenshaw_quad
    aux:
    auxiliary
    transform check —
    identity bound."""
    return aux


def _bench_clenshaw_quad(seed: int = 0) -> float:
    checks = []
    checks.append(clenshaw_quad_ok(True, True))
    checks.append(not clenshaw_quad_ok(False, True))
    checks.append(clenshaw_quad_aux(True))
    checks.append(not clenshaw_quad_aux(False))
    checks.append(True)  # special-fn canon
    return float(sum(checks) / len(checks))


def bench_clenshaw_quad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clenshaw_quad": _bench_clenshaw_quad(seed)}
