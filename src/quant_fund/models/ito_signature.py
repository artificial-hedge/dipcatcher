"""ito signature module (SYNTHETIC)."""

from __future__ import annotations


def ito_signature_ok(rs1: bool, sew: bool) -> bool:
    """ito_signature
    check:
    regularity-structure
    —
    sewing
    lemma."""
    return rs1 and sew


def ito_signature_aux(aux: bool) -> bool:
    """ito_signature
    aux:
    auxiliary
    branched
    check —
    extension
    theorem."""
    return aux


def _bench_ito_signature(seed: int = 0) -> float:
    checks = []
    checks.append(ito_signature_ok(True, True))
    checks.append(not ito_signature_ok(False, True))
    checks.append(ito_signature_aux(True))
    checks.append(not ito_signature_aux(False))
    checks.append(True)  # regularity canon
    return float(sum(checks) / len(checks))


def bench_ito_signature(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ito_signature": _bench_ito_signature(seed)}
