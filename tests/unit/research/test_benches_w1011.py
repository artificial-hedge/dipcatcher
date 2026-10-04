"""Wave-1011 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1011 import (
    bench_band_structure_family,
    bench_bloch_theorem_family,
    bench_hubbard_model_family,
    bench_kondo_effect_family,
    bench_phonon_spectrum_family,
    bench_tight_binding_family,
)

_FAMILY_BENCHES = [
    bench_bloch_theorem_family,
    bench_tight_binding_family,
    bench_phonon_spectrum_family,
    bench_band_structure_family,
    bench_hubbard_model_family,
    bench_kondo_effect_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
