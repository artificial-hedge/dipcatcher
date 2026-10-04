"""Wave-1100 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1100 import (
    bench_analytical_chemistry_family,
    bench_biochemistry_family,
    bench_electrochemistry_family,
    bench_inorganic_chemistry_family,
    bench_organic_chemistry_family,
    bench_physical_chemistry_family,
)

_FAMILY_BENCHES = [
    bench_organic_chemistry_family,
    bench_inorganic_chemistry_family,
    bench_physical_chemistry_family,
    bench_analytical_chemistry_family,
    bench_biochemistry_family,
    bench_electrochemistry_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
