"""Wave-993 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w993 import (
    bench_degree_theory_family,
    bench_krein_rutman_family,
    bench_maximal_monotone_family,
    bench_minty_browder_family,
    bench_monotone_op_family,
    bench_schauder_fixed_family,
)

_FAMILY_BENCHES = [
    bench_monotone_op_family,
    bench_degree_theory_family,
    bench_schauder_fixed_family,
    bench_krein_rutman_family,
    bench_minty_browder_family,
    bench_maximal_monotone_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
