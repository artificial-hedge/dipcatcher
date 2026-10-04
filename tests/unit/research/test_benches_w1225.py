"""Wave-1225 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1225 import (
    bench_colorectal_surgery_family,
    bench_general_surgery_studies_family,
    bench_hepatobiliary_surgery_family,
    bench_minimally_invasive_surgery_family,
    bench_surgical_oncology_studies_family,
    bench_trauma_surgery_family,
)

_FAMILY_BENCHES = [
    bench_general_surgery_studies_family,
    bench_trauma_surgery_family,
    bench_colorectal_surgery_family,
    bench_hepatobiliary_surgery_family,
    bench_surgical_oncology_studies_family,
    bench_minimally_invasive_surgery_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
