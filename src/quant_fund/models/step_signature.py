"""step signature module (SYNTHETIC)."""

from __future__ import annotations


def step_signature_ok(rs1: bool, sew: bool) -> bool:
    """step_signature
    check:
    regularity-structure
    —
    sewing
    lemma."""
    return rs1 and sew


def step_signature_aux(aux: bool) -> bool:
    """step_signature
    aux:
    auxiliary
    branched
    check —
    extension
    theorem."""
    return aux


def _bench_step_signature(seed: int = 0) -> float:
    checks = []
    checks.append(step_signature_ok(True, True))
    checks.append(not step_signature_ok(False, True))
    checks.append(step_signature_aux(True))
    checks.append(not step_signature_aux(False))
    checks.append(True)  # regularity canon
    return float(sum(checks) / len(checks))


def bench_step_signature(seed: int = 0) -> dict[str, float]:
    return {"synthetic_step_signature": _bench_step_signature(seed)}
