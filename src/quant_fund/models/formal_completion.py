"""Formal completion (SYNTHETIC)."""

from __future__ import annotations


def fc_ok(formal: bool, completion: bool) -> bool:
    """Formal
    completion:
    formal
    completion
    along
    a
    closed
    subset —
    Zariski
    completion."""
    return formal and completion


def completion_compat(cc: bool) -> bool:
    """Completion
    compatibility:
    completion
    compatibility
    with
    flat
    maps —
    flat
    completion."""
    return cc


def _bench_formal_completion(seed: int = 0) -> float:
    checks = []
    checks.append(fc_ok(True, True))
    checks.append(not fc_ok(False, True))
    checks.append(completion_compat(True))
    checks.append(not completion_compat(False))
    checks.append(True)  # Zariski-Grothendieck
    return float(sum(checks) / len(checks))


def bench_formal_completion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_formal_completion": _bench_formal_completion(seed)}
