"""Wave-1045 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1045 import (
    bench_geochemistry_family,
    bench_geochronology_family,
    bench_paleontology_family,
    bench_petrology_family,
    bench_stratigraphy_family,
    bench_structural_geology_family,
)

_FAMILY_BENCHES = [
    bench_stratigraphy_family,
    bench_structural_geology_family,
    bench_petrology_family,
    bench_geochemistry_family,
    bench_geochronology_family,
    bench_paleontology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
