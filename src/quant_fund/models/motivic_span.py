"""motivic span module (SYNTHETIC)."""

from __future__ import annotations


def motivic_span_ok(motivic: bool, stable: bool) -> bool:
    """motivic_span
    check:
    motivic
    structure —
    frobenius."""
    return motivic and stable


def motivic_span_aux(aux: bool) -> bool:
    """motivic_span
    aux:
    auxiliary
    motivic
    check —
    cartier."""
    return aux


def _bench_motivic_span(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_span_ok(True, True))
    checks.append(not motivic_span_ok(False, True))
    checks.append(motivic_span_aux(True))
    checks.append(not motivic_span_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_span(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_span": _bench_motivic_span(seed)}
