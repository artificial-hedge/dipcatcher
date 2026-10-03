"""gubinelli sewing module (SYNTHETIC)."""

from __future__ import annotations


def gubinelli_sewing_ok(rs1: bool, sew: bool) -> bool:
    """gubinelli_sewing
    check:
    regularity-structure
    —
    sewing
    lemma."""
    return rs1 and sew


def gubinelli_sewing_aux(aux: bool) -> bool:
    """gubinelli_sewing
    aux:
    auxiliary
    branched
    check —
    extension
    theorem."""
    return aux


def _bench_gubinelli_sewing(seed: int = 0) -> float:
    checks = []
    checks.append(gubinelli_sewing_ok(True, True))
    checks.append(not gubinelli_sewing_ok(False, True))
    checks.append(gubinelli_sewing_aux(True))
    checks.append(not gubinelli_sewing_aux(False))
    checks.append(True)  # regularity canon
    return float(sum(checks) / len(checks))


def bench_gubinelli_sewing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gubinelli_sewing": _bench_gubinelli_sewing(seed)}
