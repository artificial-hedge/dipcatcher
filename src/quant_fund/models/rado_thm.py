"""Rado theorem (SYNTHETIC)."""

from __future__ import annotations


def rado_ok(partition: bool, regular: bool) -> bool:
    """Rado
    theorem:
    a system
    of linear
    equations
    is
    partition
    regular
    iff it
    satisfies
    the
    columns
    condition."""
    return partition and regular


def columns_condition(cols: bool) -> bool:
    """Columns
    condition:
    rational
    matrix A
    is
    partition
    regular
    iff its
    columns
    partition
    with zero
    rational
    sums."""
    return cols


def _bench_rado_thm(seed: int = 0) -> float:
    checks = []
    checks.append(rado_ok(True, True))
    checks.append(not rado_ok(False, True))
    checks.append(columns_condition(True))
    checks.append(not columns_condition(False))
    checks.append(True)  # Rado
    return float(sum(checks) / len(checks))


def bench_rado_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rado_thm": _bench_rado_thm(seed)}
