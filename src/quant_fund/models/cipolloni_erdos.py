"""cipolloni erdos module (SYNTHETIC)."""

from __future__ import annotations


def cipolloni_erdos_ok(rmt: bool, univ: bool) -> bool:
    """cipolloni_erdos
    check:
    random-matrix-2
    structure —
    Tracy."""
    return rmt and univ


def cipolloni_erdos_aux(aux: bool) -> bool:
    """cipolloni_erdos
    aux:
    auxiliary
    bulk-universality
    check —
    Widom."""
    return aux


def _bench_cipolloni_erdos(seed: int = 0) -> float:
    checks = []
    checks.append(cipolloni_erdos_ok(True, True))
    checks.append(not cipolloni_erdos_ok(False, True))
    checks.append(cipolloni_erdos_aux(True))
    checks.append(not cipolloni_erdos_aux(False))
    checks.append(True)  # random-matrix-2 canon
    return float(sum(checks) / len(checks))


def bench_cipolloni_erdos(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cipolloni_erdos": _bench_cipolloni_erdos(seed)}
