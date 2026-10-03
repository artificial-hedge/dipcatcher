"""Wave-355 probability-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w355 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kolmogorov_axioms": b.bench_kolmogorov_axioms_family(),
        "conditional_expect": b.bench_conditional_expect_family(),
        "markov_ineq": b.bench_markov_ineq_family(),
        "conv_sum": b.bench_conv_sum_family(),
        "moment_generating": b.bench_moment_generating_family(),
        "stochastic_order": b.bench_stochastic_order_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
