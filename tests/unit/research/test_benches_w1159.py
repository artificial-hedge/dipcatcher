"""Wave-1159 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1159 import (
    bench_aerospace_engineering_2_family,
    bench_biomedical_engineering_2_family,
    bench_chemical_engineering_2_family,
    bench_civil_engineering_2_family,
    bench_electrical_engineering_2_family,
    bench_mechanical_engineering_2_family,
)

_FAMILY_BENCHES = [
    bench_biomedical_engineering_2_family,
    bench_chemical_engineering_2_family,
    bench_mechanical_engineering_2_family,
    bench_civil_engineering_2_family,
    bench_electrical_engineering_2_family,
    bench_aerospace_engineering_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
