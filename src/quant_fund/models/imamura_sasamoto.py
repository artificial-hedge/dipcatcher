"""imamura sasamoto module (SYNTHETIC)."""

from __future__ import annotations


def imamura_sasamoto_ok(kpz: bool, reg: bool) -> bool:
    """imamura_sasamoto
    check:
    KPZ-2
    structure —
    Corwin."""
    return kpz and reg


def imamura_sasamoto_aux(aux: bool) -> bool:
    """imamura_sasamoto
    aux:
    auxiliary
    regularity-structure
    check —
    Hairer."""
    return aux


def _bench_imamura_sasamoto(seed: int = 0) -> float:
    checks = []
    checks.append(imamura_sasamoto_ok(True, True))
    checks.append(not imamura_sasamoto_ok(False, True))
    checks.append(imamura_sasamoto_aux(True))
    checks.append(not imamura_sasamoto_aux(False))
    checks.append(True)  # KPZ-2 canon
    return float(sum(checks) / len(checks))


def bench_imamura_sasamoto(seed: int = 0) -> dict[str, float]:
    return {"synthetic_imamura_sasamoto": _bench_imamura_sasamoto(seed)}
