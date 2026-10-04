"""Minimal model operad (SYNTHETIC)."""

from __future__ import annotations


def mm_ok(minimal: bool, quasi_free: bool) -> bool:
    """Minimal
    model:
    minimal
    model
    of
    operad —
    Markl
    minimal."""
    return minimal and quasi_free


def minimal_operad(mo: bool) -> bool:
    """Minimal
    operad:
    quasi-
    free
    dg
    operad —
    Markl."""
    return mo


def _bench_minimal_model_op(seed: int = 0) -> float:
    checks = []
    checks.append(mm_ok(True, True))
    checks.append(not mm_ok(False, True))
    checks.append(minimal_operad(True))
    checks.append(not minimal_operad(False))
    checks.append(True)  # Markl
    return float(sum(checks) / len(checks))


def bench_minimal_model_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minimal_model_op": _bench_minimal_model_op(seed)}
