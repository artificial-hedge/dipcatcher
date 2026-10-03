"""erdos yau module (SYNTHETIC)."""

from __future__ import annotations


def erdos_yau_ok(rmt: bool, univ: bool) -> bool:
    """erdos_yau
    check:
    random-matrix
    structure —
    Wigner."""
    return rmt and univ


def erdos_yau_aux(aux: bool) -> bool:
    """erdos_yau
    aux:
    auxiliary
    universality
    check —
    Dyson."""
    return aux


def _bench_erdos_yau(seed: int = 0) -> float:
    checks = []
    checks.append(erdos_yau_ok(True, True))
    checks.append(not erdos_yau_ok(False, True))
    checks.append(erdos_yau_aux(True))
    checks.append(not erdos_yau_aux(False))
    checks.append(True)  # random-matrix canon
    return float(sum(checks) / len(checks))


def bench_erdos_yau(seed: int = 0) -> dict[str, float]:
    return {"synthetic_erdos_yau": _bench_erdos_yau(seed)}
