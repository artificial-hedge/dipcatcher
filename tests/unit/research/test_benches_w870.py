"""Wave-870 optimization adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w870 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "newton_method": b.bench_newton_method_family(),
        "quasi_newton_lbfgs": b.bench_quasi_newton_lbfgs_family(),
        "augmented_lagrangian": b.bench_augmented_lagrangian_family(),
        "interior_point2": b.bench_interior_point2_family(),
        "grad_descent_nest": b.bench_grad_descent_nest_family(),
        "conjugate_opt": b.bench_conjugate_opt_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
