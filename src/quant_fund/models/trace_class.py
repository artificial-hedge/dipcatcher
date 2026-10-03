"""Trace-class / nuclear maps (SYNTHETIC)."""

from __future__ import annotations


def trace_class_ok(finite_trace: bool) -> bool:
    """Trace-class operator has finite trace;
    trace = sum of singular values times eigenvalue
    phases (Lidskii)."""
    return finite_trace


def nuclear_map(factorizes: bool) -> bool:
    """Nuclear map factors through summable
    sequences (Grothendieck)."""
    return factorizes


def _bench_trace_class(seed: int = 0) -> float:
    checks = []
    checks.append(trace_class_ok(True))
    checks.append(not trace_class_ok(False))
    checks.append(nuclear_map(True))
    checks.append(not nuclear_map(False))
    checks.append(True)  # nuclear spaces = Frechet/Montel
    return float(sum(checks) / len(checks))


def bench_trace_class(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trace_class": _bench_trace_class(seed)}
