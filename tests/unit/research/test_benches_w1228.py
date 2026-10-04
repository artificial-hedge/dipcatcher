"""Wave-1228 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1228 import (
    bench_dental_studies_family,
    bench_endodontic_studies_family,
    bench_oral_surgery_studies_family,
    bench_orthodontic_studies_family,
    bench_pediatric_dentistry_family,
    bench_periodontal_studies_family,
)

_FAMILY_BENCHES = [
    bench_dental_studies_family,
    bench_oral_surgery_studies_family,
    bench_endodontic_studies_family,
    bench_periodontal_studies_family,
    bench_orthodontic_studies_family,
    bench_pediatric_dentistry_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
