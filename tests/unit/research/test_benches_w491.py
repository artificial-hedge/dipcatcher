"""Wave-491 automorphic-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w491 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "shimura_var": b.bench_shimura_var_family(),
        "l_function": b.bench_l_function_family(),
        "hecke_alg2": b.bench_hecke_alg2_family(),
        "theta_lift": b.bench_theta_lift_family(),
        "arthur_param": b.bench_arthur_param_family(),
        "satake_param": b.bench_satake_param_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
