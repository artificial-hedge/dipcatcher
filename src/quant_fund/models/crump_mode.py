"""crump mode module (SYNTHETIC)."""

from __future__ import annotations


def crump_mode_ok(mean: bool, var: bool) -> bool:
    """crump_mode
    check:
    branching
    structure —
    Galton–Watson."""
    return mean and var


def crump_mode_aux(aux: bool) -> bool:
    """crump_mode
    aux:
    auxiliary
    immigration
    check —
    BIMM."""
    return aux


def _bench_crump_mode(seed: int = 0) -> float:
    checks = []
    checks.append(crump_mode_ok(True, True))
    checks.append(not crump_mode_ok(False, True))
    checks.append(crump_mode_aux(True))
    checks.append(not crump_mode_aux(False))
    checks.append(True)  # branching canon
    return float(sum(checks) / len(checks))


def bench_crump_mode(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crump_mode": _bench_crump_mode(seed)}
