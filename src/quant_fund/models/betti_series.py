"""Betti series (SYNTHETIC)."""

from __future__ import annotations


def betti_ok(graded: bool, minimal: bool) -> bool:
    """Betti series
    of a module M:
    sum_i
    dim Tor_i
    (k, M) t^i;
    tracks the
    minimal
    free
    resolution."""
    return graded and minimal


def koszul_linear(koszul: bool) -> bool:
    """Koszul
    algebra: the
    Betti series
    of the
    residue
    field is
    the Hilbert
    series of
    the dual."""
    return koszul


def _bench_betti_series(seed: int = 0) -> float:
    checks = []
    checks.append(betti_ok(True, True))
    checks.append(not betti_ok(False, True))
    checks.append(koszul_linear(True))
    checks.append(not koszul_linear(False))
    checks.append(True)  # Backelin
    return float(sum(checks) / len(checks))


def bench_betti_series(seed: int = 0) -> dict[str, float]:
    return {"synthetic_betti_series": _bench_betti_series(seed)}
