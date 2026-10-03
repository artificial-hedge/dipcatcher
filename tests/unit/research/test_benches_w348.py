"""Wave-348 graph-theory adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w348 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "graph_coloring": b.bench_graph_coloring_family(),
        "euler_trail": b.bench_euler_trail_family(),
        "matroid_greedy": b.bench_matroid_greedy_family(),
        "planar_check": b.bench_planar_check_family(),
        "poset_dimension": b.bench_poset_dimension_family(),
        "ramsey_r33": b.bench_ramsey_r33_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
