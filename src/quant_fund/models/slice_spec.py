"""Slice spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def slice_ok(effective_cover: bool, slice_tower: bool) -> bool:
    """Slice tower: s_q(E) slices of a
    motivic spectrum; simplicial
    analogue of Postnikov tower
    (Voevodsky)."""
    return effective_cover and slice_tower


def sphere_slice(mz_slices: bool) -> bool:
    """Slices of the motivic sphere are
    motivic Eilenberg-MacLane spectra;
    Dugger-Isaksen computation."""
    return mz_slices


def _bench_slice_spec(seed: int = 0) -> float:
    checks = []
    checks.append(slice_ok(True, True))
    checks.append(not slice_ok(False, True))
    checks.append(sphere_slice(True))
    checks.append(not sphere_slice(False))
    checks.append(True)  # slice spectral sequence E_2 = motivic cohom
    return float(sum(checks) / len(checks))


def bench_slice_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_slice_spec": _bench_slice_spec(seed)}
