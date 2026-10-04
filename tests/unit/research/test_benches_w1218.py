"""Wave-1218 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1218 import (
    bench_acid_base_medicine_family,
    bench_dialysis_medicine_family,
    bench_hypertension_medicine_family,
    bench_nephrology_studies_family,
    bench_renal_transplant_family,
    bench_urology_studies_family,
)

_FAMILY_BENCHES = [
    bench_nephrology_studies_family,
    bench_dialysis_medicine_family,
    bench_renal_transplant_family,
    bench_acid_base_medicine_family,
    bench_hypertension_medicine_family,
    bench_urology_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
