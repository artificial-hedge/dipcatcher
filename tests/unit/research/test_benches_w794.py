"""Wave-794 stochastic-control adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w794 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dynamic_programming": b.bench_dynamic_programming_family(),
        "verification_thm": b.bench_verification_thm_family(),
        "hamilton_jacobi": b.bench_hamilton_jacobi_family(),
        "viscosity_solution": b.bench_viscosity_solution_family(),
        "quasi_variational": b.bench_quasi_variational_family(),
        "impulsive_control": b.bench_impulsive_control_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
