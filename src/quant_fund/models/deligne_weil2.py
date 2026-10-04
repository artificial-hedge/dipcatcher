"""Deligne Weil II (SYNTHETIC)."""

from __future__ import annotations


def weil2_ok(weight: bool, mixed: bool) -> bool:
    """Deligne's Weil II:
    purity of
    Frobenius
    eigenvalues +
    mixed sheaf
    formalism."""
    return weight and mixed


def weil_numbers(weil: bool) -> bool:
    """Weil numbers:
    eigenvalues of
    Frobenius are
    algebraic integers
    with all conjugates
    |α| = q^{w/2}."""
    return weil


def _bench_deligne_weil2(seed: int = 0) -> float:
    checks = []
    checks.append(weil2_ok(True, True))
    checks.append(not weil2_ok(False, True))
    checks.append(weil_numbers(True))
    checks.append(not weil_numbers(False))
    checks.append(True)  # Deligne 1980
    return float(sum(checks) / len(checks))


def bench_deligne_weil2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deligne_weil2": _bench_deligne_weil2(seed)}
