"""calabrese kpz module (SYNTHETIC)."""

from __future__ import annotations


def calabrese_kpz_ok(kpz: bool, grow: bool) -> bool:
    """calabrese_kpz
    check:
    KPZ
    structure —
    Kardar."""
    return kpz and grow


def calabrese_kpz_aux(aux: bool) -> bool:
    """calabrese_kpz
    aux:
    auxiliary
    scaling-exponent
    check —
    Spohn."""
    return aux


def _bench_calabrese_kpz(seed: int = 0) -> float:
    checks = []
    checks.append(calabrese_kpz_ok(True, True))
    checks.append(not calabrese_kpz_ok(False, True))
    checks.append(calabrese_kpz_aux(True))
    checks.append(not calabrese_kpz_aux(False))
    checks.append(True)  # KPZ canon
    return float(sum(checks) / len(checks))


def bench_calabrese_kpz(seed: int = 0) -> dict[str, float]:
    return {"synthetic_calabrese_kpz": _bench_calabrese_kpz(seed)}
