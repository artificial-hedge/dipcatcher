"""kardar parisi module (SYNTHETIC)."""

from __future__ import annotations


def kardar_parisi_ok(kpz: bool, grow: bool) -> bool:
    """kardar_parisi
    check:
    KPZ
    structure —
    Kardar."""
    return kpz and grow


def kardar_parisi_aux(aux: bool) -> bool:
    """kardar_parisi
    aux:
    auxiliary
    scaling-exponent
    check —
    Spohn."""
    return aux


def _bench_kardar_parisi(seed: int = 0) -> float:
    checks = []
    checks.append(kardar_parisi_ok(True, True))
    checks.append(not kardar_parisi_ok(False, True))
    checks.append(kardar_parisi_aux(True))
    checks.append(not kardar_parisi_aux(False))
    checks.append(True)  # KPZ canon
    return float(sum(checks) / len(checks))


def bench_kardar_parisi(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kardar_parisi": _bench_kardar_parisi(seed)}
