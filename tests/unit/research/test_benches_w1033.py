"""Wave-1033 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1033 import (
    bench_bioinstrumentation_family,
    bench_biomechanics_family,
    bench_biomedical_imaging2_family,
    bench_medical_devices_family,
    bench_physiological_modeling_family,
    bench_tissue_engineering_family,
)

_FAMILY_BENCHES = [
    bench_biomechanics_family,
    bench_medical_devices_family,
    bench_tissue_engineering_family,
    bench_bioinstrumentation_family,
    bench_physiological_modeling_family,
    bench_biomedical_imaging2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
