"""Wave-1193 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1193 import (
    bench_audiology_studies_family,
    bench_clinical_psychology_2_family,
    bench_midwifery_studies_family,
    bench_opticianry_family,
    bench_orthoptics_family,
    bench_prosthetics_orthotics_family,
)

_FAMILY_BENCHES = [
    bench_midwifery_studies_family,
    bench_orthoptics_family,
    bench_audiology_studies_family,
    bench_opticianry_family,
    bench_prosthetics_orthotics_family,
    bench_clinical_psychology_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
