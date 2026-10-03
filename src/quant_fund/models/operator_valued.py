"""Operator-valued free probability (SYNTHETIC)."""

from __future__ import annotations


def ov_ok(bimodule: bool, conditional_expectation: bool) -> bool:
    """Operator-
    valued:
    free
    probability
    over
    a
    subalgebra —
    conditional
    expectation
    replacing
    scalar
    state."""
    return bimodule and conditional_expectation


def operator_r_transform(ort: bool) -> bool:
    """Operator
    R-
    transform:
    blockwise
    free
    probability —
    matrix
    models
    with
    structure."""
    return ort


def _bench_operator_valued(seed: int = 0) -> float:
    checks = []
    checks.append(ov_ok(True, True))
    checks.append(not ov_ok(False, True))
    checks.append(operator_r_transform(True))
    checks.append(not operator_r_transform(False))
    checks.append(True)  # Speicher
    return float(sum(checks) / len(checks))


def bench_operator_valued(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operator_valued": _bench_operator_valued(seed)}
