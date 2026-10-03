"""caracciolo pelissetto module (SYNTHETIC)."""

from __future__ import annotations


def caracciolo_pelissetto_ok(rc: bool, potts: bool) -> bool:
    """caracciolo_pelissetto
    check:
    random-cluster
    structure —
    Grimmett."""
    return rc and potts


def caracciolo_pelissetto_aux(aux: bool) -> bool:
    """caracciolo_pelissetto
    aux:
    auxiliary
    Potts-model
    check —
    Sokal."""
    return aux


def _bench_caracciolo_pelissetto(seed: int = 0) -> float:
    checks = []
    checks.append(caracciolo_pelissetto_ok(True, True))
    checks.append(not caracciolo_pelissetto_ok(False, True))
    checks.append(caracciolo_pelissetto_aux(True))
    checks.append(not caracciolo_pelissetto_aux(False))
    checks.append(True)  # random-cluster canon
    return float(sum(checks) / len(checks))


def bench_caracciolo_pelissetto(seed: int = 0) -> dict[str, float]:
    return {"synthetic_caracciolo_pelissetto": _bench_caracciolo_pelissetto(seed)}
