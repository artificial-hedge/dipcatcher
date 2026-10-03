"""Wave-978 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w978 import (
    bench_bochner_riesz_family,
    bench_hausdorff_young_family,
    bench_lp_multiplier_family,
    bench_oscillatory_int_family,
    bench_restriction_est_family,
    bench_strichartz_est_family,
)

_FAMILY_BENCHES = [
    bench_hausdorff_young_family,
    bench_restriction_est_family,
    bench_bochner_riesz_family,
    bench_lp_multiplier_family,
    bench_oscillatory_int_family,
    bench_strichartz_est_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
