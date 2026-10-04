"""Wave-1117 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1117 import (
    bench_historical_sociology_family,
    bench_legal_sociology_family,
    bench_mathematical_sociology_family,
    bench_military_sociology_family,
    bench_science_studies_family,
    bench_sociology_of_knowledge_family,
)

_FAMILY_BENCHES = [
    bench_mathematical_sociology_family,
    bench_historical_sociology_family,
    bench_science_studies_family,
    bench_sociology_of_knowledge_family,
    bench_military_sociology_family,
    bench_legal_sociology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
