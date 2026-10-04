"""Wave-560 geometric-flows adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w560 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hamilton_ricci": b.bench_hamilton_ricci_family(),
        "perelman_entropy": b.bench_perelman_entropy_family(),
        "ricci_soliton": b.bench_ricci_soliton_family(),
        "kahler_ricci_flow": b.bench_kahler_ricci_flow_family(),
        "mean_curvature_flow": b.bench_mean_curvature_flow_family(),
        "ancient_solution": b.bench_ancient_solution_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
