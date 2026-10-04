"""Wave-1016 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1016 import (
    bench_coherence_theory_family,
    bench_diffraction_grating_family,
    bench_fourier_optics_family,
    bench_holography_family,
    bench_interference_fringes_family,
    bench_polarization_states_family,
)

_FAMILY_BENCHES = [
    bench_diffraction_grating_family,
    bench_fourier_optics_family,
    bench_interference_fringes_family,
    bench_polarization_states_family,
    bench_coherence_theory_family,
    bench_holography_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
