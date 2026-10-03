"""Wave-971 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w971 import (
    bench_connes_metric_family,
    bench_differential_form_nc_family,
    bench_geodesic_nc_family,
    bench_hochschild_cycle_family,
    bench_index_pairing_family,
    bench_spectral_triple_family,
)

_FAMILY_BENCHES = [
    bench_spectral_triple_family,
    bench_connes_metric_family,
    bench_index_pairing_family,
    bench_hochschild_cycle_family,
    bench_differential_form_nc_family,
    bench_geodesic_nc_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
