"""Wave-314 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w314 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "nurbs_eval": b.bench_nurbs_eval_family(),
        "catmull_clark": b.bench_catmull_clark_family(),
        "loop_subdiv": b.bench_loop_subdiv_family(),
        "marching_cubes": b.bench_marching_cubes_family(),
        "half_edge": b.bench_half_edge_family(),
        "laplacian_smooth": b.bench_laplacian_smooth_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
