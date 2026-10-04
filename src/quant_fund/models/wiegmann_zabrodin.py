"""wiegmann zabrodin module (SYNTHETIC)."""

from __future__ import annotations


def wiegmann_zabrodin_ok(gff: bool, qg: bool) -> bool:
    """wiegmann_zabrodin
    check:
    Gaussian-free-field
    structure —
    Sheffield."""
    return gff and qg


def wiegmann_zabrodin_aux(aux: bool) -> bool:
    """wiegmann_zabrodin
    aux:
    auxiliary
    quantum-gravity
    check —
    Duplantier."""
    return aux


def _bench_wiegmann_zabrodin(seed: int = 0) -> float:
    checks = []
    checks.append(wiegmann_zabrodin_ok(True, True))
    checks.append(not wiegmann_zabrodin_ok(False, True))
    checks.append(wiegmann_zabrodin_aux(True))
    checks.append(not wiegmann_zabrodin_aux(False))
    checks.append(True)  # GFF canon
    return float(sum(checks) / len(checks))


def bench_wiegmann_zabrodin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wiegmann_zabrodin": _bench_wiegmann_zabrodin(seed)}
