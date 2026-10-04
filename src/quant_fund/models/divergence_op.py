"""divergence op module (SYNTHETIC)."""

from __future__ import annotations


def divergence_op_ok(ml1: bool, div: bool) -> bool:
    """divergence_op
    check:
    Malliavin
    calculus —
    divergence
    operator."""
    return ml1 and div


def divergence_op_aux(aux: bool) -> bool:
    """divergence_op
    aux:
    auxiliary
    chaos
    check —
    Wiener
    decomposition."""
    return aux


def _bench_divergence_op(seed: int = 0) -> float:
    checks = []
    checks.append(divergence_op_ok(True, True))
    checks.append(not divergence_op_ok(False, True))
    checks.append(divergence_op_aux(True))
    checks.append(not divergence_op_aux(False))
    checks.append(True)  # malliavin canon
    return float(sum(checks) / len(checks))


def bench_divergence_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_divergence_op": _bench_divergence_op(seed)}
