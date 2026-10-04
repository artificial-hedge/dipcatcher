"""Wave-1145 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1145 import (
    bench_bioinformatics_4_family,
    bench_epidemiology_3_family,
    bench_genomicsciences_family,
    bench_proteomics_family,
    bench_synthetic_biology_family,
    bench_systems_biology_2_family,
)

_FAMILY_BENCHES = [
    bench_genomicsciences_family,
    bench_proteomics_family,
    bench_bioinformatics_4_family,
    bench_systems_biology_2_family,
    bench_synthetic_biology_family,
    bench_epidemiology_3_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
