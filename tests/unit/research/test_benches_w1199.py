"""Wave-1199 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1199 import (
    bench_cardiac_electrophysiology_family,
    bench_dialysis_technology_family,
    bench_hepatobiliary_studies_family,
    bench_interventional_radiology_family,
    bench_nuclear_cardiology_family,
    bench_transplant_studies_family,
)

_FAMILY_BENCHES = [
    bench_dialysis_technology_family,
    bench_transplant_studies_family,
    bench_hepatobiliary_studies_family,
    bench_cardiac_electrophysiology_family,
    bench_interventional_radiology_family,
    bench_nuclear_cardiology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
