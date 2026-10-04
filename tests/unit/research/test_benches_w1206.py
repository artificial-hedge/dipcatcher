"""Wave-1206 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1206 import (
    bench_immunosuppression_family,
    bench_organ_donation_family,
    bench_regenerative_medicine_family,
    bench_stem_cell_therapy_family,
    bench_transplantation_medicine_family,
    bench_xenotransplantation_family,
)

_FAMILY_BENCHES = [
    bench_transplantation_medicine_family,
    bench_organ_donation_family,
    bench_immunosuppression_family,
    bench_xenotransplantation_family,
    bench_stem_cell_therapy_family,
    bench_regenerative_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
