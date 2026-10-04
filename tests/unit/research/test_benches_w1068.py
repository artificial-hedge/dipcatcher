"""Wave-1068 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1068 import (
    bench_conflict_resolution_family,
    bench_defense_studies_family,
    bench_intelligence_studies_family,
    bench_military_science_family,
    bench_peace_studies_family,
    bench_strategic_studies_family,
)

_FAMILY_BENCHES = [
    bench_military_science_family,
    bench_defense_studies_family,
    bench_strategic_studies_family,
    bench_intelligence_studies_family,
    bench_peace_studies_family,
    bench_conflict_resolution_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
