"""Wave-1136 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1136 import (
    bench_environmental_law_family,
    bench_evidence_law_family,
    bench_family_law_family,
    bench_immigration_law_family,
    bench_labor_law_family,
    bench_tax_law_family,
)

_FAMILY_BENCHES = [
    bench_environmental_law_family,
    bench_family_law_family,
    bench_labor_law_family,
    bench_tax_law_family,
    bench_evidence_law_family,
    bench_immigration_law_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
