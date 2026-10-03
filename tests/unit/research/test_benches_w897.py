"""Wave-897 ODE-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w897 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "linear_multistep": b.bench_linear_multistep_family(),
        "dahlquist_test": b.bench_dahlquist_test_family(),
        "explicit_midpoint": b.bench_explicit_midpoint_family(),
        "heun_method": b.bench_heun_method_family(),
        "trapezoid_rule": b.bench_trapezoid_rule_family(),
        "order_barrier": b.bench_order_barrier_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
