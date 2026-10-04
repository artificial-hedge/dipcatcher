"""Wave-1061 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1061 import (
    bench_administrative_law_family,
    bench_constitutional_law_family,
    bench_contract_law_family,
    bench_criminal_law_family,
    bench_international_law_family,
    bench_tort_law_family,
)

_FAMILY_BENCHES = [
    bench_constitutional_law_family,
    bench_criminal_law_family,
    bench_contract_law_family,
    bench_tort_law_family,
    bench_administrative_law_family,
    bench_international_law_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
