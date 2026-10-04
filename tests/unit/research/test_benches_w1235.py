"""Wave-1235 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1235 import (
    bench_connective_tissue_studies_family,
    bench_inflammatory_arthritis_studies_family,
    bench_myositis_studies_family,
    bench_osteoarthritis_studies_family,
    bench_rheumatology_medicine_family,
    bench_spondyloarthritis_studies_family,
)

_FAMILY_BENCHES = [
    bench_rheumatology_medicine_family,
    bench_spondyloarthritis_studies_family,
    bench_inflammatory_arthritis_studies_family,
    bench_connective_tissue_studies_family,
    bench_osteoarthritis_studies_family,
    bench_myositis_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
