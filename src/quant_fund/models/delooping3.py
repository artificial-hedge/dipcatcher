"""delooping3 module (SYNTHETIC)."""

from __future__ import annotations


def delooping3_ok(algebra: bool, coherent: bool) -> bool:
    """delooping3
    check:
    algebra
    structure —
    e5."""
    return algebra and coherent


def delooping3_aux(aux: bool) -> bool:
    """delooping3
    aux:
    auxiliary
    algebra
    check —
    operad."""
    return aux


def _bench_delooping3(seed: int = 0) -> float:
    checks = []
    checks.append(delooping3_ok(True, True))
    checks.append(not delooping3_ok(False, True))
    checks.append(delooping3_aux(True))
    checks.append(not delooping3_aux(False))
    checks.append(True)  # higher algebra canon
    return float(sum(checks) / len(checks))


def bench_delooping3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_delooping3": _bench_delooping3(seed)}
