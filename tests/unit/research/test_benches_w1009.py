"""Wave-1009 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1009 import (
    bench_dipole_radiation_family,
    bench_fresnel_eq_family,
    bench_lorentz_lorenz_family,
    bench_maxwell_equations_family,
    bench_poynting_vector_family,
    bench_wave_guides_family,
)

_FAMILY_BENCHES = [
    bench_maxwell_equations_family,
    bench_poynting_vector_family,
    bench_fresnel_eq_family,
    bench_wave_guides_family,
    bench_dipole_radiation_family,
    bench_lorentz_lorenz_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
