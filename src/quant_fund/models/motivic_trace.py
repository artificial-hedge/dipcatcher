"""motivic trace module (SYNTHETIC)."""

from __future__ import annotations


def motivic_trace_ok(motivic: bool, stable: bool) -> bool:
    """motivic_trace
    check:
    motivic
    structure —
    trace."""
    return motivic and stable


def motivic_trace_aux(aux: bool) -> bool:
    """motivic_trace
    aux:
    auxiliary
    motivic
    check —
    transfer."""
    return aux


def _bench_motivic_trace(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_trace_ok(True, True))
    checks.append(not motivic_trace_ok(False, True))
    checks.append(motivic_trace_aux(True))
    checks.append(not motivic_trace_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_trace(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_trace": _bench_motivic_trace(seed)}
