"""Wave-962 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w962 import (
    bench_double_commutant_family,
    bench_jones_index_family,
    bench_normal_state_family,
    bench_predual_space_family,
    bench_tomita_takesaki_family,
    bench_von_neumann_alg_family,
)

_FAMILY_BENCHES = [
    bench_von_neumann_alg_family,
    bench_double_commutant_family,
    bench_predual_space_family,
    bench_normal_state_family,
    bench_tomita_takesaki_family,
    bench_jones_index_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
