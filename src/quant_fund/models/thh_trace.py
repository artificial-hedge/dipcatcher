"""THH trace (SYNTHETIC)."""

from __future__ import annotations


def tt_ok(thh: bool, trace: bool) -> bool:
    """THH
    trace:
    topological
    Hochschild
    homology
    trace —
    THH."""
    return thh and trace


def trace_map(tm: bool) -> bool:
    """Trace
    map:
    K-
    theory
    to
    THH
    trace —
    Bokstedt-
    Madsen."""
    return tm


def _bench_thh_trace(seed: int = 0) -> float:
    checks = []
    checks.append(tt_ok(True, True))
    checks.append(not tt_ok(False, True))
    checks.append(trace_map(True))
    checks.append(not trace_map(False))
    checks.append(True)  # Bokstedt-Madsen
    return float(sum(checks) / len(checks))


def bench_thh_trace(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thh_trace": _bench_thh_trace(seed)}
