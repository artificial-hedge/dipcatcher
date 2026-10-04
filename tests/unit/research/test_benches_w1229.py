"""Wave-1229 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1229 import (
    bench_adolescent_medicine_studies_family,
    bench_developmental_pediatrics_family,
    bench_neonatal_medicine_studies_family,
    bench_pediatric_cardiology_family,
    bench_pediatric_oncology_family,
    bench_pediatrics_studies_family,
)

_FAMILY_BENCHES = [
    bench_pediatrics_studies_family,
    bench_neonatal_medicine_studies_family,
    bench_pediatric_cardiology_family,
    bench_pediatric_oncology_family,
    bench_adolescent_medicine_studies_family,
    bench_developmental_pediatrics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
