"""Wave-1014 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1014 import (
    bench_cmb_anisotropy_family,
    bench_dark_matter_family,
    bench_hubble_law_family,
    bench_jeans_instability_family,
    bench_stellar_evolution_family,
    bench_stellar_structure_family,
)

_FAMILY_BENCHES = [
    bench_jeans_instability_family,
    bench_stellar_structure_family,
    bench_stellar_evolution_family,
    bench_hubble_law_family,
    bench_cmb_anisotropy_family,
    bench_dark_matter_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
