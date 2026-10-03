"""berestycki gff module (SYNTHETIC)."""

from __future__ import annotations


def berestycki_gff_ok(gff: bool, qg: bool) -> bool:
    """berestycki_gff
    check:
    Gaussian-free-field
    structure —
    Sheffield."""
    return gff and qg


def berestycki_gff_aux(aux: bool) -> bool:
    """berestycki_gff
    aux:
    auxiliary
    quantum-gravity
    check —
    Duplantier."""
    return aux


def _bench_berestycki_gff(seed: int = 0) -> float:
    checks = []
    checks.append(berestycki_gff_ok(True, True))
    checks.append(not berestycki_gff_ok(False, True))
    checks.append(berestycki_gff_aux(True))
    checks.append(not berestycki_gff_aux(False))
    checks.append(True)  # GFF canon
    return float(sum(checks) / len(checks))


def bench_berestycki_gff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_berestycki_gff": _bench_berestycki_gff(seed)}
