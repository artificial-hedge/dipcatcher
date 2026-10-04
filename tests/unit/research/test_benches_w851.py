"""Wave-851 discontinuous-Galerkin adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w851 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dg_discretization": b.bench_dg_discretization_family(),
        "numerical_flux_dg": b.bench_numerical_flux_dg_family(),
        "penalty_dg": b.bench_penalty_dg_family(),
        "modal_basis": b.bench_modal_basis_family(),
        "limiter_tvb": b.bench_limiter_tvb_family(),
        "rkdg_step": b.bench_rkdg_step_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
