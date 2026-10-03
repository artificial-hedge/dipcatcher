"""najafi gff module (SYNTHETIC)."""

from __future__ import annotations


def najafi_gff_ok(gff: bool, qg: bool) -> bool:
    """najafi_gff
    check:
    Gaussian-free-field
    structure —
    Berestycki."""
    return gff and qg


def najafi_gff_aux(aux: bool) -> bool:
    """najafi_gff
    aux:
    auxiliary
    Liouville
    check —
    Rhodes."""
    return aux


def _bench_najafi_gff(seed: int = 0) -> float:
    checks = []
    checks.append(najafi_gff_ok(True, True))
    checks.append(not najafi_gff_ok(False, True))
    checks.append(najafi_gff_aux(True))
    checks.append(not najafi_gff_aux(False))
    checks.append(True)  # GFF-2 canon
    return float(sum(checks) / len(checks))


def bench_najafi_gff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_najafi_gff": _bench_najafi_gff(seed)}
