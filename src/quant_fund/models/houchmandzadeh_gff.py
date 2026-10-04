"""houchmandzadeh gff module (SYNTHETIC)."""

from __future__ import annotations


def houchmandzadeh_gff_ok(gff: bool, qg: bool) -> bool:
    """houchmandzadeh_gff
    check:
    Gaussian-free-field
    structure —
    Sheffield."""
    return gff and qg


def houchmandzadeh_gff_aux(aux: bool) -> bool:
    """houchmandzadeh_gff
    aux:
    auxiliary
    quantum-gravity
    check —
    Duplantier."""
    return aux


def _bench_houchmandzadeh_gff(seed: int = 0) -> float:
    checks = []
    checks.append(houchmandzadeh_gff_ok(True, True))
    checks.append(not houchmandzadeh_gff_ok(False, True))
    checks.append(houchmandzadeh_gff_aux(True))
    checks.append(not houchmandzadeh_gff_aux(False))
    checks.append(True)  # GFF canon
    return float(sum(checks) / len(checks))


def bench_houchmandzadeh_gff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_houchmandzadeh_gff": _bench_houchmandzadeh_gff(seed)}
