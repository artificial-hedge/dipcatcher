"""Wave-865 a-posteriori error-estimation adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w865 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "residual_estimator": b.bench_residual_estimator_family(),
        "zienkiewicz_zhu": b.bench_zienkiewicz_zhu_family(),
        "recovery_error": b.bench_recovery_error_family(),
        "dual_weighted_res": b.bench_dual_weighted_res_family(),
        "goal_oriented": b.bench_goal_oriented_family(),
        "equilibrated_flux": b.bench_equilibrated_flux_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
