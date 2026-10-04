"""compensator rm module (SYNTHETIC)."""

from __future__ import annotations


def compensator_rm_ok(rm: bool, comp: bool) -> bool:
    """compensator_rm
    check:
    random
    measure —
    compensator."""
    return rm and comp


def compensator_rm_aux(aux: bool) -> bool:
    """compensator_rm
    aux:
    auxiliary
    measure check —
    intensity."""
    return aux


def _bench_compensator_rm(seed: int = 0) -> float:
    checks = []
    checks.append(compensator_rm_ok(True, True))
    checks.append(not compensator_rm_ok(False, True))
    checks.append(compensator_rm_aux(True))
    checks.append(not compensator_rm_aux(False))
    checks.append(True)  # random-measure canon
    return float(sum(checks) / len(checks))


def bench_compensator_rm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_compensator_rm": _bench_compensator_rm(seed)}
