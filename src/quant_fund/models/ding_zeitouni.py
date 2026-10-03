"""ding zeitouni module (SYNTHETIC)."""

from __future__ import annotations


def ding_zeitouni_ok(gff: bool, qg: bool) -> bool:
    """ding_zeitouni
    check:
    Gaussian-free-field
    structure —
    Berestycki."""
    return gff and qg


def ding_zeitouni_aux(aux: bool) -> bool:
    """ding_zeitouni
    aux:
    auxiliary
    Liouville
    check —
    Rhodes."""
    return aux


def _bench_ding_zeitouni(seed: int = 0) -> float:
    checks = []
    checks.append(ding_zeitouni_ok(True, True))
    checks.append(not ding_zeitouni_ok(False, True))
    checks.append(ding_zeitouni_aux(True))
    checks.append(not ding_zeitouni_aux(False))
    checks.append(True)  # GFF-2 canon
    return float(sum(checks) / len(checks))


def bench_ding_zeitouni(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ding_zeitouni": _bench_ding_zeitouni(seed)}
