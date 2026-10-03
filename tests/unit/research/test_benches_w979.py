"""Wave-979 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w979 import (
    bench_disjointness_dyn_family,
    bench_horocycle_flow_family,
    bench_ratner_thm_family,
    bench_unipotent_ergodic_family,
    bench_van_der_corput_family,
    bench_weyl_equidist_family,
)

_FAMILY_BENCHES = [
    bench_weyl_equidist_family,
    bench_van_der_corput_family,
    bench_horocycle_flow_family,
    bench_unipotent_ergodic_family,
    bench_ratner_thm_family,
    bench_disjointness_dyn_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
