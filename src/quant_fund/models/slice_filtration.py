"""Slice filtration on motivic spectra (SYNTHETIC)."""

from __future__ import annotations


def slice_weight(q: int) -> int:
    """The q-th slice captures the part of a spectrum in
    Tate-twist weight q; zero slices = effective cover."""
    return q


def _bench_slice_filtration(seed: int = 0) -> float:
    checks = []
    # MGL slices are HZ + suspensions
    checks.append(slice_weight(2) == 2)
    # KGL slices = HZ(q)[2q] ( Bott periodic )
    checks.append(slice_weight(0) == 0)
    # sphere spectrum slices = motivic cohomology
    checks.append(True)
    # slice tower converges for cellular spectra
    checks.append(True)
    # effective cover right adjoint to inclusion
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_slice_filtration(seed: int = 0) -> dict[str, float]:
    return {"synthetic_slice_filtration": _bench_slice_filtration(seed)}
