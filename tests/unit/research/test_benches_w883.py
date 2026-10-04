"""Wave-883 acceleration adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w883 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "tangent_predictor": b.bench_tangent_predictor_family(),
        "is_drift": b.bench_is_drift_family(),
        "cv_optimal": b.bench_cv_optimal_family(),
        "nest_accel": b.bench_nest_accel_family(),
        "subgradient_descent": b.bench_subgradient_descent_family(),
        "min_var_closure": b.bench_min_var_closure_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
