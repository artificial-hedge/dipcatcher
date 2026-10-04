"""Wave-1052 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1052 import (
    bench_clinical_nutrition_family,
    bench_dietary_assessment_family,
    bench_metabolic_health_family,
    bench_nutritional_biochemistry_family,
    bench_nutritional_epidemiology_family,
    bench_sports_nutrition_family,
)

_FAMILY_BENCHES = [
    bench_nutritional_biochemistry_family,
    bench_dietary_assessment_family,
    bench_clinical_nutrition_family,
    bench_sports_nutrition_family,
    bench_nutritional_epidemiology_family,
    bench_metabolic_health_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
