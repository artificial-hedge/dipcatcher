"""Wave-858 adaptive/oscillatory-quadrature adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w858 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "adaptive_simpsons": b.bench_adaptive_simpsons_family(),
        "tanh_sinh": b.bench_tanh_sinh_family(),
        "double_exp_quad": b.bench_double_exp_quad_family(),
        "osc_singular": b.bench_osc_singular_family(),
        "filon_quad": b.bench_filon_quad_family(),
        "levin_quad": b.bench_levin_quad_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
