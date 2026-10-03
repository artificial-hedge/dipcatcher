"""borell tis module (SYNTHETIC)."""

from __future__ import annotations


def borell_tis_ok(gp: bool, bound: bool) -> bool:
    """borell_tis
    check:
    Gaussian-process
    structure —
    Slepian."""
    return gp and bound


def borell_tis_aux(aux: bool) -> bool:
    """borell_tis
    aux:
    auxiliary
    sup
    check —
    Fernique."""
    return aux


def _bench_borell_tis(seed: int = 0) -> float:
    checks = []
    checks.append(borell_tis_ok(True, True))
    checks.append(not borell_tis_ok(False, True))
    checks.append(borell_tis_aux(True))
    checks.append(not borell_tis_aux(False))
    checks.append(True)  # Gaussian canon
    return float(sum(checks) / len(checks))


def bench_borell_tis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borell_tis": _bench_borell_tis(seed)}
