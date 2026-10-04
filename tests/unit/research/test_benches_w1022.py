"""Wave-1022 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1022 import (
    bench_auction_theory2_family,
    bench_growth_theory_family,
    bench_mechanism_design_family,
    bench_overlapping_gens_family,
    bench_real_business_family,
    bench_search_matching_family,
)

_FAMILY_BENCHES = [
    bench_growth_theory_family,
    bench_overlapping_gens_family,
    bench_real_business_family,
    bench_search_matching_family,
    bench_mechanism_design_family,
    bench_auction_theory2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
