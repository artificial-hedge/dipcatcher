"""Wave-942 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w942 import (
    bench_backward_forward_family,
    bench_ishikawa_iter_family,
    bench_malitsky_golden_family,
    bench_mann_iter_family,
    bench_primal_dual_hybrid_family,
    bench_vu_condat_family,
)

_FAMILY_BENCHES = [
    bench_primal_dual_hybrid_family,
    bench_vu_condat_family,
    bench_backward_forward_family,
    bench_malitsky_golden_family,
    bench_mann_iter_family,
    bench_ishikawa_iter_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
