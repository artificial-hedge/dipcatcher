"""Wave-1188 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1188 import (
    bench_brewing_science_family,
    bench_culinary_science_family,
    bench_enology_family,
    bench_fermentation_studies_family,
    bench_gastronomy_2_family,
    bench_pastry_arts_family,
)

_FAMILY_BENCHES = [
    bench_culinary_science_family,
    bench_pastry_arts_family,
    bench_brewing_science_family,
    bench_enology_family,
    bench_fermentation_studies_family,
    bench_gastronomy_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
