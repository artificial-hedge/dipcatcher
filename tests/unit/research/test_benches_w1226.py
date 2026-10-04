"""Wave-1226 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1226 import (
    bench_anatomical_pathology_family,
    bench_clinical_pathology_family,
    bench_cytopathology_family,
    bench_histopathology_studies_family,
    bench_molecular_pathology_family,
    bench_pathology_studies_family,
)

_FAMILY_BENCHES = [
    bench_pathology_studies_family,
    bench_anatomical_pathology_family,
    bench_clinical_pathology_family,
    bench_histopathology_studies_family,
    bench_cytopathology_family,
    bench_molecular_pathology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
