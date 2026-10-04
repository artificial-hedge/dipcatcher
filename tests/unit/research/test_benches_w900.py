"""Wave-900 spatial-index adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w900 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kd_tree": b.bench_kd_tree_family(),
        "ball_tree": b.bench_ball_tree_family(),
        "cover_tree": b.bench_cover_tree_family(),
        "r_tree": b.bench_r_tree_family(),
        "quad_tree": b.bench_quad_tree_family(),
        "vp_tree": b.bench_vp_tree_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
