"""yosida op module (SYNTHETIC)."""

from __future__ import annotations


def yosida_op_ok(sc: bool, sp: bool) -> bool:
    """yosida_op
    check:
    diffusion
    theory —
    boundary."""
    return sc and sp


def yosida_op_aux(aux: bool) -> bool:
    """yosida_op
    aux:
    auxiliary
    diffusion
    check —
    generator."""
    return aux


def _bench_yosida_op(seed: int = 0) -> float:
    checks = []
    checks.append(yosida_op_ok(True, True))
    checks.append(not yosida_op_ok(False, True))
    checks.append(yosida_op_aux(True))
    checks.append(not yosida_op_aux(False))
    checks.append(True)  # diffusion canon
    return float(sum(checks) / len(checks))


def bench_yosida_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_yosida_op": _bench_yosida_op(seed)}
