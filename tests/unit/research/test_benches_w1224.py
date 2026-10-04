"""Wave-1224 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1224 import (
    bench_fetal_medicine_family,
    bench_gynecologic_oncology_family,
    bench_gynecology_studies_family,
    bench_maternal_fetal_medicine_family,
    bench_obstetrics_studies_family,
    bench_reproductive_endocrinology_family,
)

_FAMILY_BENCHES = [
    bench_obstetrics_studies_family,
    bench_gynecology_studies_family,
    bench_maternal_fetal_medicine_family,
    bench_reproductive_endocrinology_family,
    bench_gynecologic_oncology_family,
    bench_fetal_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
