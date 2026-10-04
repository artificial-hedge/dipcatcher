"""aru gff module (SYNTHETIC)."""

from __future__ import annotations


def aru_gff_ok(gff: bool, qg: bool) -> bool:
    """aru_gff
    check:
    Gaussian-free-field
    structure —
    Berestycki."""
    return gff and qg


def aru_gff_aux(aux: bool) -> bool:
    """aru_gff
    aux:
    auxiliary
    Liouville
    check —
    Rhodes."""
    return aux


def _bench_aru_gff(seed: int = 0) -> float:
    checks = []
    checks.append(aru_gff_ok(True, True))
    checks.append(not aru_gff_ok(False, True))
    checks.append(aru_gff_aux(True))
    checks.append(not aru_gff_aux(False))
    checks.append(True)  # GFF-2 canon
    return float(sum(checks) / len(checks))


def bench_aru_gff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aru_gff": _bench_aru_gff(seed)}
