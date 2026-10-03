"""Wave-1004 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1004 import (
    bench_bgk_model_family,
    bench_boltzmann_eq_family,
    bench_chapman_enskog_family,
    bench_h_theorem_family,
    bench_landau_damping_family,
    bench_vlasov_eq_family,
)

_FAMILY_BENCHES = [
    bench_boltzmann_eq_family,
    bench_vlasov_eq_family,
    bench_bgk_model_family,
    bench_chapman_enskog_family,
    bench_h_theorem_family,
    bench_landau_damping_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
