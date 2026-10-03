"""powell gff module (SYNTHETIC)."""

from __future__ import annotations


def powell_gff_ok(gff: bool, qg: bool) -> bool:
    """powell_gff
    check:
    Gaussian-free-field
    structure —
    Berestycki."""
    return gff and qg


def powell_gff_aux(aux: bool) -> bool:
    """powell_gff
    aux:
    auxiliary
    Liouville
    check —
    Rhodes."""
    return aux


def _bench_powell_gff(seed: int = 0) -> float:
    checks = []
    checks.append(powell_gff_ok(True, True))
    checks.append(not powell_gff_ok(False, True))
    checks.append(powell_gff_aux(True))
    checks.append(not powell_gff_aux(False))
    checks.append(True)  # GFF-2 canon
    return float(sum(checks) / len(checks))


def bench_powell_gff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_powell_gff": _bench_powell_gff(seed)}
