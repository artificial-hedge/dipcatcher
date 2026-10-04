"""Wave-875 adaptive-marking adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w875 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "adaptive_marking": b.bench_adaptive_marking_family(),
        "hierarchical_est": b.bench_hierarchical_est_family(),
        "dorfler_marking": b.bench_dorfler_marking_family(),
        "convergence_theory": b.bench_convergence_theory_family(),
        "adaptive_finite": b.bench_adaptive_finite_family(),
        "goal_adaptive": b.bench_goal_adaptive_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
