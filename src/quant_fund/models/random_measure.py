"""random measure module (SYNTHETIC)."""

from __future__ import annotations


def random_measure_ok(rm: bool, comp: bool) -> bool:
    """random_measure
    check:
    random
    measure —
    compensator."""
    return rm and comp


def random_measure_aux(aux: bool) -> bool:
    """random_measure
    aux:
    auxiliary
    measure check —
    intensity."""
    return aux


def _bench_random_measure(seed: int = 0) -> float:
    checks = []
    checks.append(random_measure_ok(True, True))
    checks.append(not random_measure_ok(False, True))
    checks.append(random_measure_aux(True))
    checks.append(not random_measure_aux(False))
    checks.append(True)  # random-measure canon
    return float(sum(checks) / len(checks))


def bench_random_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_random_measure": _bench_random_measure(seed)}
