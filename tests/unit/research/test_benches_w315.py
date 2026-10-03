"""Wave-315 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w315 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "rmpflow": b.bench_rmpflow_family(),
        "ds_motion": b.bench_ds_motion_family(),
        "wbc_qp": b.bench_wbc_qp_family(),
        "grasp_epsilon": b.bench_grasp_epsilon_family(),
        "rrt_connect": b.bench_rrt_connect_family(),
        "dmp_control": b.bench_dmp_control_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
