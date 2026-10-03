"""Wave-869 MC-variance-reduction adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w869 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "antithetic_var": b.bench_antithetic_var_family(),
        "control_variate": b.bench_control_variate_family(),
        "importance_sampling": b.bench_importance_sampling_family(),
        "stratified_var": b.bench_stratified_var_family(),
        "common_random": b.bench_common_random_family(),
        "conditional_mc": b.bench_conditional_mc_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
