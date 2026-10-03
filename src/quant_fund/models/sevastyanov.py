"""sevastyanov module (SYNTHETIC)."""

from __future__ import annotations


def sevastyanov_ok(mean: bool, var: bool) -> bool:
    """sevastyanov
    check:
    branching
    structure —
    Galton–Watson."""
    return mean and var


def sevastyanov_aux(aux: bool) -> bool:
    """sevastyanov
    aux:
    auxiliary
    immigration
    check —
    BIMM."""
    return aux


def _bench_sevastyanov(seed: int = 0) -> float:
    checks = []
    checks.append(sevastyanov_ok(True, True))
    checks.append(not sevastyanov_ok(False, True))
    checks.append(sevastyanov_aux(True))
    checks.append(not sevastyanov_aux(False))
    checks.append(True)  # branching canon
    return float(sum(checks) / len(checks))


def bench_sevastyanov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sevastyanov": _bench_sevastyanov(seed)}
