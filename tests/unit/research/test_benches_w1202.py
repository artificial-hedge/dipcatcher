"""Wave-1202 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1202 import (
    bench_entomology_medical_family,
    bench_medical_microbiology_family,
    bench_mycology_studies_family,
    bench_parasitology_studies_family,
    bench_public_health_microbiology_family,
    bench_vector_borne_diseases_family,
)

_FAMILY_BENCHES = [
    bench_public_health_microbiology_family,
    bench_medical_microbiology_family,
    bench_parasitology_studies_family,
    bench_mycology_studies_family,
    bench_entomology_medical_family,
    bench_vector_borne_diseases_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
