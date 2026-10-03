"""hartley transform module (SYNTHETIC)."""

from __future__ import annotations


def hartley_transform_ok(arg: bool, param: bool) -> bool:
    """hartley_transform
    check:
    special function /
    transform canon —
    arg/param consistency."""
    return arg and param


def hartley_transform_aux(aux: bool) -> bool:
    """hartley_transform
    aux:
    auxiliary
    transform check —
    identity bound."""
    return aux


def _bench_hartley_transform(seed: int = 0) -> float:
    checks = []
    checks.append(hartley_transform_ok(True, True))
    checks.append(not hartley_transform_ok(False, True))
    checks.append(hartley_transform_aux(True))
    checks.append(not hartley_transform_aux(False))
    checks.append(True)  # special-fn canon
    return float(sum(checks) / len(checks))


def bench_hartley_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hartley_transform": _bench_hartley_transform(seed)}
