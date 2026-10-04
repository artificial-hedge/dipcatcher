"""Wave-1162 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1162 import (
    bench_criminology_2_family,
    bench_international_relations_2_family,
    bench_law_5_family,
    bench_military_science_2_family,
    bench_political_science_4_family,
    bench_public_administration_2_family,
)

_FAMILY_BENCHES = [
    bench_law_5_family,
    bench_political_science_4_family,
    bench_public_administration_2_family,
    bench_international_relations_2_family,
    bench_criminology_2_family,
    bench_military_science_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
