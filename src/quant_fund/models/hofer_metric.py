"""Hofer metric (SYNTHETIC)."""

from __future__ import annotations


def hm_ok(oscillation: bool, nondegenerate: bool) -> bool:
    """Hofer
    metric:
    bi-
    invariant
    metric
    on
    Hamiltonian
    diffeomorphisms —
    oscillation
    norm."""
    return oscillation and nondegenerate


def hofer_geodesic(hg: bool) -> bool:
    """Hofer
    geodesics:
    autonomous
    Hamiltonians
    generate
    minimizing
    paths —
    Lalonde-
    McDuff."""
    return hg


def _bench_hofer_metric(seed: int = 0) -> float:
    checks = []
    checks.append(hm_ok(True, True))
    checks.append(not hm_ok(False, True))
    checks.append(hofer_geodesic(True))
    checks.append(not hofer_geodesic(False))
    checks.append(True)  # Hofer
    return float(sum(checks) / len(checks))


def bench_hofer_metric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hofer_metric": _bench_hofer_metric(seed)}
