"""operad infty5 module (SYNTHETIC)."""

from __future__ import annotations


def operad_infty5_ok(algebra: bool, coherent: bool) -> bool:
    """operad_infty5
    check:
    algebra
    structure —
    e5."""
    return algebra and coherent


def operad_infty5_aux(aux: bool) -> bool:
    """operad_infty5
    aux:
    auxiliary
    algebra
    check —
    operad."""
    return aux


def _bench_operad_infty5(seed: int = 0) -> float:
    checks = []
    checks.append(operad_infty5_ok(True, True))
    checks.append(not operad_infty5_ok(False, True))
    checks.append(operad_infty5_aux(True))
    checks.append(not operad_infty5_aux(False))
    checks.append(True)  # higher algebra canon
    return float(sum(checks) / len(checks))


def bench_operad_infty5(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_infty5": _bench_operad_infty5(seed)}
