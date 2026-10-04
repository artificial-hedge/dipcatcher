"""Wave-1201 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1201 import (
    bench_biomedical_informatics_family,
    bench_clinical_informatics_family,
    bench_health_data_science_family,
    bench_health_informatics_family,
    bench_health_information_family,
    bench_medical_records_family,
)

_FAMILY_BENCHES = [
    bench_health_informatics_family,
    bench_medical_records_family,
    bench_health_information_family,
    bench_biomedical_informatics_family,
    bench_clinical_informatics_family,
    bench_health_data_science_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
