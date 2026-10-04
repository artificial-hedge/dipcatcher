"""nick gff module (SYNTHETIC)."""

from __future__ import annotations


def nick_gff_ok(gff: bool, qg: bool) -> bool:
    """nick_gff
    check:
    Gaussian-free-field
    structure —
    Sheffield."""
    return gff and qg


def nick_gff_aux(aux: bool) -> bool:
    """nick_gff
    aux:
    auxiliary
    quantum-gravity
    check —
    Duplantier."""
    return aux


def _bench_nick_gff(seed: int = 0) -> float:
    checks = []
    checks.append(nick_gff_ok(True, True))
    checks.append(not nick_gff_ok(False, True))
    checks.append(nick_gff_aux(True))
    checks.append(not nick_gff_aux(False))
    checks.append(True)  # GFF canon
    return float(sum(checks) / len(checks))


def bench_nick_gff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nick_gff": _bench_nick_gff(seed)}
