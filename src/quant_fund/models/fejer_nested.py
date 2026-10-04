"""fejer nested module (SYNTHETIC)."""

from __future__ import annotations


def fejer_nested_ok(arg: bool, param: bool) -> bool:
    """fejer_nested
    check:
    special function /
    transform canon —
    arg/param consistency."""
    return arg and param


def fejer_nested_aux(aux: bool) -> bool:
    """fejer_nested
    aux:
    auxiliary
    transform check —
    identity bound."""
    return aux


def _bench_fejer_nested(seed: int = 0) -> float:
    checks = []
    checks.append(fejer_nested_ok(True, True))
    checks.append(not fejer_nested_ok(False, True))
    checks.append(fejer_nested_aux(True))
    checks.append(not fejer_nested_aux(False))
    checks.append(True)  # special-fn canon
    return float(sum(checks) / len(checks))


def bench_fejer_nested(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fejer_nested": _bench_fejer_nested(seed)}
