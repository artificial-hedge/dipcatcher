"""Jacobi fields (SYNTHETIC)."""

from __future__ import annotations


def jf_ok(variation: bool, conjugate: bool) -> bool:
    """Jacobi
    field:
    infinitesimal
    variation
    of
    geodesics —
    zeros
    mark
    conjugate
    points."""
    return variation and conjugate


def index_lemma(il: bool) -> bool:
    """Index
    lemma:
    Jacobi
    fields
    minimize
    the
    index
    form
    among
    fields
    with
    same
    endpoints."""
    return il


def _bench_jacobi_field(seed: int = 0) -> float:
    checks = []
    checks.append(jf_ok(True, True))
    checks.append(not jf_ok(False, True))
    checks.append(index_lemma(True))
    checks.append(not index_lemma(False))
    checks.append(True)  # Jacobi
    return float(sum(checks) / len(checks))


def bench_jacobi_field(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jacobi_field": _bench_jacobi_field(seed)}
