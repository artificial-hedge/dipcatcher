"""Wave-1203 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1203 import (
    bench_genomic_medicine_family,
    bench_laboratory_medicine_family,
    bench_molecular_diagnostics_family,
    bench_precision_medicine_family,
    bench_travel_medicine_family,
    bench_tropical_medicine_family,
)

_FAMILY_BENCHES = [
    bench_tropical_medicine_family,
    bench_travel_medicine_family,
    bench_genomic_medicine_family,
    bench_precision_medicine_family,
    bench_molecular_diagnostics_family,
    bench_laboratory_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
