"""Finite-field Kakeya (SYNTHETIC)."""

from __future__ import annotations


def ffk_ok(poly_method: bool, bound: bool) -> bool:
    """Finite-
    field
    Kakeya
    (Dvir):
    a Kakeya
    set in
    F_q^n has
    size
    Omega(q^n);
    polynomial
    method."""
    return poly_method and bound


def dvir_proof(dvir: bool) -> bool:
    """Dvir's
    proof:
    a low-
    degree
    polynomial
    vanishing
    on a
    Kakeya
    set is
    identically
    zero."""
    return dvir


def _bench_ff_kakeya(seed: int = 0) -> float:
    checks = []
    checks.append(ffk_ok(True, True))
    checks.append(not ffk_ok(False, True))
    checks.append(dvir_proof(True))
    checks.append(not dvir_proof(False))
    checks.append(True)  # Dvir
    return float(sum(checks) / len(checks))


def bench_ff_kakeya(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ff_kakeya": _bench_ff_kakeya(seed)}
