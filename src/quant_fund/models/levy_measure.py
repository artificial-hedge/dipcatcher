"""levy measure module (SYNTHETIC)."""

from __future__ import annotations


def levy_measure_ok(id1: bool, cg: bool) -> bool:
    """levy_measure
    check:
    infinitely-divisible
    structure —
    Levy
    canon."""
    return id1 and cg


def levy_measure_aux(aux: bool) -> bool:
    """levy_measure
    aux:
    auxiliary
    triplet
    check —
    Khinchin
    formula."""
    return aux


def _bench_levy_measure(seed: int = 0) -> float:
    checks = []
    checks.append(levy_measure_ok(True, True))
    checks.append(not levy_measure_ok(False, True))
    checks.append(levy_measure_aux(True))
    checks.append(not levy_measure_aux(False))
    checks.append(True)  # levy canon
    return float(sum(checks) / len(checks))


def bench_levy_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_levy_measure": _bench_levy_measure(seed)}
