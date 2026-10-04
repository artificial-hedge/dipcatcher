"""Wave-1236 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1236 import (
    bench_chronic_pain_studies_family,
    bench_fibromyalgia_studies_family,
    bench_headache_studies_family,
    bench_interventional_pain_studies_family,
    bench_neuropathic_pain_studies_family,
    bench_opioid_stewardship_studies_family,
)

_FAMILY_BENCHES = [
    bench_chronic_pain_studies_family,
    bench_fibromyalgia_studies_family,
    bench_headache_studies_family,
    bench_neuropathic_pain_studies_family,
    bench_opioid_stewardship_studies_family,
    bench_interventional_pain_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
