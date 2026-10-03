"""Wave-311 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w311 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fm_partition": b.bench_fm_partition_family(),
        "lee_router": b.bench_lee_router_family(),
        "clock_tree": b.bench_clock_tree_family(),
        "aig_rewrite": b.bench_aig_rewrite_family(),
        "power_est": b.bench_power_est_family(),
        "floorplan_sa": b.bench_floorplan_sa_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
