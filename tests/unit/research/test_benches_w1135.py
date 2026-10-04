"""Wave-1135 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1135 import (
    bench_ophthalmology_family,
    bench_otolaryngology_family,
    bench_palliative_medicine_family,
    bench_rehabilitation_medicine_family,
    bench_sports_medicine_family,
    bench_urology_family,
)

_FAMILY_BENCHES = [
    bench_urology_family,
    bench_ophthalmology_family,
    bench_otolaryngology_family,
    bench_palliative_medicine_family,
    bench_sports_medicine_family,
    bench_rehabilitation_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
