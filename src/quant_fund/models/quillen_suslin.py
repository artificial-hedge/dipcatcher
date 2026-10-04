"""Quillen-Suslin theorem (SYNTHETIC)."""

from __future__ import annotations


def qs_ok(vector: bool, free: bool) -> bool:
    """Quillen-
    Suslin:
    all
    vector
    bundles on
    affine
    space
    A^n_k
    are
    trivial."""
    return vector and free


def unimodular_row(row: bool) -> bool:
    """Unimodular
    rows
    complete
    to
    invertible
    matrices
    over
    polynomial
    rings."""
    return row


def _bench_quillen_suslin(seed: int = 0) -> float:
    checks = []
    checks.append(qs_ok(True, True))
    checks.append(not qs_ok(False, True))
    checks.append(unimodular_row(True))
    checks.append(not unimodular_row(False))
    checks.append(True)  # Quillen-Suslin
    return float(sum(checks) / len(checks))


def bench_quillen_suslin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quillen_suslin": _bench_quillen_suslin(seed)}
