"""Wave-1148 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1148 import (
    bench_geochronology_2_family,
    bench_geology_3_family,
    bench_geomorphology_2_family,
    bench_mineralogy_2_family,
    bench_petrology_2_family,
    bench_stratigraphy_2_family,
)

_FAMILY_BENCHES = [
    bench_geology_3_family,
    bench_petrology_2_family,
    bench_mineralogy_2_family,
    bench_stratigraphy_2_family,
    bench_geomorphology_2_family,
    bench_geochronology_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
