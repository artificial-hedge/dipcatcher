"""Operad modules (SYNTHETIC)."""

from __future__ import annotations


def om_ok(operad: bool, module: bool) -> bool:
    """Operad
    module:
    module
    over
    an
    operad —
    operadic
    module."""
    return operad and module


def algebra_over_op(aoo: bool) -> bool:
    """Algebra
    over:
    algebra
    over
    an
    operad —
    operadic
    algebra."""
    return aoo


def _bench_operad_module(seed: int = 0) -> float:
    checks = []
    checks.append(om_ok(True, True))
    checks.append(not om_ok(False, True))
    checks.append(algebra_over_op(True))
    checks.append(not algebra_over_op(False))
    checks.append(True)  # May
    return float(sum(checks) / len(checks))


def bench_operad_module(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_module": _bench_operad_module(seed)}
