"""Wave-1169 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1169 import (
    bench_disability_studies_2_family,
    bench_ethnic_studies_2_family,
    bench_gender_studies_2_family,
    bench_public_policy_2_family,
    bench_social_work_2_family,
    bench_urban_studies_2_family,
)

_FAMILY_BENCHES = [
    bench_social_work_2_family,
    bench_public_policy_2_family,
    bench_urban_studies_2_family,
    bench_gender_studies_2_family,
    bench_ethnic_studies_2_family,
    bench_disability_studies_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
