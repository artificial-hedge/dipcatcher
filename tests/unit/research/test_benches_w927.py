"""Wave-927 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w927 import (
    bench_ebanch_diverge_family,
    bench_expectation_param_family,
    bench_fisher_metric2_family,
    bench_potential_fn_family,
    bench_renyi_div_family,
    bench_shannon_gibbs_family,
)

_FAMILY_BENCHES = [
    bench_fisher_metric2_family,
    bench_expectation_param_family,
    bench_potential_fn_family,
    bench_ebanch_diverge_family,
    bench_shannon_gibbs_family,
    bench_renyi_div_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
