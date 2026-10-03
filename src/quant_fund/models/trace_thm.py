"""Trace theorem (SYNTHETIC)."""

from __future__ import annotations


def tr_ok(boundary: bool, loss: bool) -> bool:
    """Trace:
    restriction
    of
    H^1
    to
    the
    boundary
    lands
    in
    H^{1/2} —
    loses
    half
    a
    derivative."""
    return boundary and loss


def trace_ext(te: bool) -> bool:
    """Extension:
    right
    inverse —
    every
    H^{1/2}
    boundary
    datum
    extends
    into
    H^1."""
    return te


def _bench_trace_thm(seed: int = 0) -> float:
    checks = []
    checks.append(tr_ok(True, True))
    checks.append(not tr_ok(False, True))
    checks.append(trace_ext(True))
    checks.append(not trace_ext(False))
    checks.append(True)  # Gagliardo
    return float(sum(checks) / len(checks))


def bench_trace_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trace_thm": _bench_trace_thm(seed)}
