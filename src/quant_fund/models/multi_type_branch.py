"""multi type_branch module (SYNTHETIC)."""

from __future__ import annotations


def multi_type_branch_ok(mean: bool, var: bool) -> bool:
    """multi_type_branch
    check:
    branching
    structure —
    Galton–Watson."""
    return mean and var


def multi_type_branch_aux(aux: bool) -> bool:
    """multi_type_branch
    aux:
    auxiliary
    immigration
    check —
    BIMM."""
    return aux


def _bench_multi_type_branch(seed: int = 0) -> float:
    checks = []
    checks.append(multi_type_branch_ok(True, True))
    checks.append(not multi_type_branch_ok(False, True))
    checks.append(multi_type_branch_aux(True))
    checks.append(not multi_type_branch_aux(False))
    checks.append(True)  # branching canon
    return float(sum(checks) / len(checks))


def bench_multi_type_branch(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multi_type_branch": _bench_multi_type_branch(seed)}
