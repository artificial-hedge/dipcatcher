"""Wave-1138 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1138 import (
    bench_behavioral_economics_family,
    bench_econ_neuroscience_family,
    bench_evolutionary_economics_family,
    bench_experimental_economics_2_family,
    bench_institutional_economics_family,
    bench_political_economy_2_family,
)

_FAMILY_BENCHES = [
    bench_behavioral_economics_family,
    bench_econ_neuroscience_family,
    bench_experimental_economics_2_family,
    bench_institutional_economics_family,
    bench_evolutionary_economics_family,
    bench_political_economy_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
