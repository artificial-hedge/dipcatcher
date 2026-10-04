"""Wave-1160 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1160 import (
    bench_area_studies_2_family,
    bench_classics_2_family,
    bench_history_5_family,
    bench_humanities_2_family,
    bench_philosophy_6_family,
    bench_religious_studies_2_family,
)

_FAMILY_BENCHES = [
    bench_philosophy_6_family,
    bench_history_5_family,
    bench_religious_studies_2_family,
    bench_classics_2_family,
    bench_area_studies_2_family,
    bench_humanities_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
