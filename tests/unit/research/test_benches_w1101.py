"""Wave-1101 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1101 import (
    bench_botany_family,
    bench_cell_biology_family,
    bench_genetics_family,
    bench_microbiology_family,
    bench_molecular_biology_family,
    bench_zoology_family,
)

_FAMILY_BENCHES = [
    bench_molecular_biology_family,
    bench_cell_biology_family,
    bench_genetics_family,
    bench_microbiology_family,
    bench_zoology_family,
    bench_botany_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
