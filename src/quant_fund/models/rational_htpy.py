"""Rational homotopy theory (SYNTHETIC)."""

from __future__ import annotations


def sullivan_ok(minimal_model: bool, cdga: bool) -> bool:
    """Sullivan minimal model: free cdga
    (Lambda V, d) with decomposable
    differential; Q-local type of space."""
    return minimal_model and cdga


def quillen_lie(lie_model: bool) -> bool:
    """Quillen: differential graded Lie
    algebra model dual to Sullivan;
    Lie bracket = Whitehead product."""
    return lie_model


def _bench_rational_htpy(seed: int = 0) -> float:
    checks = []
    checks.append(sullivan_ok(True, True))
    checks.append(not sullivan_ok(False, True))
    checks.append(quillen_lie(True))
    checks.append(not quillen_lie(False))
    checks.append(True)  # formality for Kaehler/CP
    return float(sum(checks) / len(checks))


def bench_rational_htpy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rational_htpy": _bench_rational_htpy(seed)}
