"""Wave-1152 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1152 import (
    bench_biochemistry_2_family,
    bench_cell_biology_2_family,
    bench_genetics_2_family,
    bench_molecular_biology_2_family,
    bench_pharmacology_2_family,
    bench_toxicology_3_family,
)

_FAMILY_BENCHES = [
    bench_biochemistry_2_family,
    bench_molecular_biology_2_family,
    bench_cell_biology_2_family,
    bench_genetics_2_family,
    bench_pharmacology_2_family,
    bench_toxicology_3_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
