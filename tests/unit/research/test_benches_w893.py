"""Wave-893 collocation adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w893 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "covello_est": b.bench_covello_est_family(),
        "dual_goal_est": b.bench_dual_goal_est_family(),
        "pseudospectral_coll": b.bench_pseudospectral_coll_family(),
        "tau_method": b.bench_tau_method_family(),
        "galerkin_least_sq": b.bench_galerkin_least_sq_family(),
        "coarsening_mark": b.bench_coarsening_mark_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
