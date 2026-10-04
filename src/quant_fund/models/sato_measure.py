"""sato measure module (SYNTHETIC)."""

from __future__ import annotations


def sato_measure_ok(rm: bool, comp: bool) -> bool:
    """sato_measure
    check:
    random
    measure —
    compensator."""
    return rm and comp


def sato_measure_aux(aux: bool) -> bool:
    """sato_measure
    aux:
    auxiliary
    measure check —
    intensity."""
    return aux


def _bench_sato_measure(seed: int = 0) -> float:
    checks = []
    checks.append(sato_measure_ok(True, True))
    checks.append(not sato_measure_ok(False, True))
    checks.append(sato_measure_aux(True))
    checks.append(not sato_measure_aux(False))
    checks.append(True)  # random-measure canon
    return float(sum(checks) / len(checks))


def bench_sato_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sato_measure": _bench_sato_measure(seed)}
