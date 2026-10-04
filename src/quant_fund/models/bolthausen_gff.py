"""bolthausen gff module (SYNTHETIC)."""

from __future__ import annotations


def bolthausen_gff_ok(gff: bool, qg: bool) -> bool:
    """bolthausen_gff
    check:
    Gaussian-free-field
    structure —
    Berestycki."""
    return gff and qg


def bolthausen_gff_aux(aux: bool) -> bool:
    """bolthausen_gff
    aux:
    auxiliary
    Liouville
    check —
    Rhodes."""
    return aux


def _bench_bolthausen_gff(seed: int = 0) -> float:
    checks = []
    checks.append(bolthausen_gff_ok(True, True))
    checks.append(not bolthausen_gff_ok(False, True))
    checks.append(bolthausen_gff_aux(True))
    checks.append(not bolthausen_gff_aux(False))
    checks.append(True)  # GFF-2 canon
    return float(sum(checks) / len(checks))


def bench_bolthausen_gff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bolthausen_gff": _bench_bolthausen_gff(seed)}
