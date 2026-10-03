"""Wave-924 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w924 import (
    bench_beta_bernoulli_family,
    bench_crp_table_family,
    bench_dp_mm_family,
    bench_gem_distribution_family,
    bench_gibbs_type_family,
    bench_neutral_process_family,
)

_FAMILY_BENCHES = [
    bench_gem_distribution_family,
    bench_dp_mm_family,
    bench_crp_table_family,
    bench_beta_bernoulli_family,
    bench_neutral_process_family,
    bench_gibbs_type_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
