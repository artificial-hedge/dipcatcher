"""Wave-1227 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1227 import (
    bench_body_imaging_family,
    bench_diagnostic_imaging_family,
    bench_interventional_neuroradiology_family,
    bench_musculoskeletal_imaging_family,
    bench_pediatric_imaging_family,
    bench_radiology_studies_family,
)

_FAMILY_BENCHES = [
    bench_radiology_studies_family,
    bench_diagnostic_imaging_family,
    bench_interventional_neuroradiology_family,
    bench_pediatric_imaging_family,
    bench_musculoskeletal_imaging_family,
    bench_body_imaging_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
