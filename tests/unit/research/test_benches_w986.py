"""Wave-986 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w986 import (
    bench_ap_weight_family,
    bench_calderon_zygmund_family,
    bench_cotlar_ineq_family,
    bench_cz_decomp_family,
    bench_good_lambda_family,
    bench_reverse_holder_family,
)

_FAMILY_BENCHES = [
    bench_calderon_zygmund_family,
    bench_cz_decomp_family,
    bench_cotlar_ineq_family,
    bench_good_lambda_family,
    bench_ap_weight_family,
    bench_reverse_holder_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
