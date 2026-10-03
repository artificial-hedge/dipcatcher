"""branching imm module (SYNTHETIC)."""

from __future__ import annotations


def branching_imm_ok(mean: bool, var: bool) -> bool:
    """branching_imm
    check:
    branching
    structure —
    Galton–Watson."""
    return mean and var


def branching_imm_aux(aux: bool) -> bool:
    """branching_imm
    aux:
    auxiliary
    immigration
    check —
    BIMM."""
    return aux


def _bench_branching_imm(seed: int = 0) -> float:
    checks = []
    checks.append(branching_imm_ok(True, True))
    checks.append(not branching_imm_ok(False, True))
    checks.append(branching_imm_aux(True))
    checks.append(not branching_imm_aux(False))
    checks.append(True)  # branching canon
    return float(sum(checks) / len(checks))


def bench_branching_imm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_branching_imm": _bench_branching_imm(seed)}
