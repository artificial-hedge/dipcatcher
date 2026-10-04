"""Wave-1187 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1187 import (
    bench_apparel_studies_family,
    bench_costume_design_family,
    bench_fashion_studies_family,
    bench_footwear_design_family,
    bench_jewelry_design_family,
    bench_textile_studies_family,
)

_FAMILY_BENCHES = [
    bench_fashion_studies_family,
    bench_textile_studies_family,
    bench_costume_design_family,
    bench_jewelry_design_family,
    bench_footwear_design_family,
    bench_apparel_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
