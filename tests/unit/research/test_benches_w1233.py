"""Wave-1233 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1233 import (
    bench_dysmorphology_studies_family,
    bench_genetic_diagnostics_family,
    bench_lysosomal_medicine_family,
    bench_medical_genetics_studies_family,
    bench_mitochondrial_medicine_family,
    bench_pharmacogenomics_studies_family,
)

_FAMILY_BENCHES = [
    bench_medical_genetics_studies_family,
    bench_genetic_diagnostics_family,
    bench_lysosomal_medicine_family,
    bench_mitochondrial_medicine_family,
    bench_dysmorphology_studies_family,
    bench_pharmacogenomics_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
