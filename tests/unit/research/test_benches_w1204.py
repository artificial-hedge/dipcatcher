"""Wave-1204 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1204 import (
    bench_aging_research_family,
    bench_geriatric_medicine_family,
    bench_gerontology_studies_family,
    bench_hospice_care_family,
    bench_longevity_medicine_family,
    bench_palliative_care_family,
)

_FAMILY_BENCHES = [
    bench_geriatric_medicine_family,
    bench_palliative_care_family,
    bench_hospice_care_family,
    bench_gerontology_studies_family,
    bench_aging_research_family,
    bench_longevity_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
