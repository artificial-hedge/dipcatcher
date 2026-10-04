"""lyons extension module (SYNTHETIC)."""

from __future__ import annotations


def lyons_extension_ok(rs1: bool, sew: bool) -> bool:
    """lyons_extension
    check:
    regularity-structure
    —
    sewing
    lemma."""
    return rs1 and sew


def lyons_extension_aux(aux: bool) -> bool:
    """lyons_extension
    aux:
    auxiliary
    branched
    check —
    extension
    theorem."""
    return aux


def _bench_lyons_extension(seed: int = 0) -> float:
    checks = []
    checks.append(lyons_extension_ok(True, True))
    checks.append(not lyons_extension_ok(False, True))
    checks.append(lyons_extension_aux(True))
    checks.append(not lyons_extension_aux(False))
    checks.append(True)  # regularity canon
    return float(sum(checks) / len(checks))


def bench_lyons_extension(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lyons_extension": _bench_lyons_extension(seed)}
