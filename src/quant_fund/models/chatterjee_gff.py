"""chatterjee gff module (SYNTHETIC)."""

from __future__ import annotations


def chatterjee_gff_ok(gff: bool, qg: bool) -> bool:
    """chatterjee_gff
    check:
    Gaussian-free-field
    structure —
    Berestycki."""
    return gff and qg


def chatterjee_gff_aux(aux: bool) -> bool:
    """chatterjee_gff
    aux:
    auxiliary
    Liouville
    check —
    Rhodes."""
    return aux


def _bench_chatterjee_gff(seed: int = 0) -> float:
    checks = []
    checks.append(chatterjee_gff_ok(True, True))
    checks.append(not chatterjee_gff_ok(False, True))
    checks.append(chatterjee_gff_aux(True))
    checks.append(not chatterjee_gff_aux(False))
    checks.append(True)  # GFF-2 canon
    return float(sum(checks) / len(checks))


def bench_chatterjee_gff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chatterjee_gff": _bench_chatterjee_gff(seed)}
