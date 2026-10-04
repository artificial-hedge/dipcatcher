"""Wave-1217 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1217 import (
    bench_adrenal_medicine_family,
    bench_bone_metabolism_family,
    bench_diabetes_medicine_family,
    bench_endocrinology_studies_family,
    bench_metabolic_medicine_family,
    bench_thyroid_medicine_family,
)

_FAMILY_BENCHES = [
    bench_endocrinology_studies_family,
    bench_diabetes_medicine_family,
    bench_thyroid_medicine_family,
    bench_metabolic_medicine_family,
    bench_bone_metabolism_family,
    bench_adrenal_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
