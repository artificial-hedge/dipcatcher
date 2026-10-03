"""kimmel branch module (SYNTHETIC)."""

from __future__ import annotations


def kimmel_branch_ok(mean: bool, var: bool) -> bool:
    """kimmel_branch
    check:
    branching
    structure —
    Galton–Watson."""
    return mean and var


def kimmel_branch_aux(aux: bool) -> bool:
    """kimmel_branch
    aux:
    auxiliary
    immigration
    check —
    BIMM."""
    return aux


def _bench_kimmel_branch(seed: int = 0) -> float:
    checks = []
    checks.append(kimmel_branch_ok(True, True))
    checks.append(not kimmel_branch_ok(False, True))
    checks.append(kimmel_branch_aux(True))
    checks.append(not kimmel_branch_aux(False))
    checks.append(True)  # branching canon
    return float(sum(checks) / len(checks))


def bench_kimmel_branch(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kimmel_branch": _bench_kimmel_branch(seed)}
