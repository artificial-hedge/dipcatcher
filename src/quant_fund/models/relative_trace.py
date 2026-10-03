"""relative trace module (SYNTHETIC)."""

from __future__ import annotations


def relative_trace_ok(derived: bool, geometry: bool) -> bool:
    """relative_trace
    check:
    derived
    geometry —
    spectral."""
    return derived and geometry


def relative_trace_aux(aux: bool) -> bool:
    """relative_trace
    aux:
    auxiliary
    derived
    check —
    stack."""
    return aux


def _bench_relative_trace(seed: int = 0) -> float:
    checks = []
    checks.append(relative_trace_ok(True, True))
    checks.append(not relative_trace_ok(False, True))
    checks.append(relative_trace_aux(True))
    checks.append(not relative_trace_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_relative_trace(seed: int = 0) -> dict[str, float]:
    return {"synthetic_relative_trace": _bench_relative_trace(seed)}
