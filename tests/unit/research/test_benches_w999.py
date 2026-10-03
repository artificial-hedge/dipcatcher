"""Wave-999 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w999 import (
    bench_brakke_varifolds_family,
    bench_currents_theory_family,
    bench_flat_chains_family,
    bench_integral_currents_family,
    bench_rectifiable_measures_family,
    bench_varifold_theory_family,
)

_FAMILY_BENCHES = [
    bench_currents_theory_family,
    bench_varifold_theory_family,
    bench_flat_chains_family,
    bench_integral_currents_family,
    bench_rectifiable_measures_family,
    bench_brakke_varifolds_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
