"""Wave-1086 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1086 import (
    bench_assyriology_family,
    bench_egyptology_family,
    bench_indology_family,
    bench_iranian_studies_family,
    bench_ottoman_studies_family,
    bench_sinology_family,
)

_FAMILY_BENCHES = [
    bench_assyriology_family,
    bench_egyptology_family,
    bench_sinology_family,
    bench_indology_family,
    bench_iranian_studies_family,
    bench_ottoman_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
