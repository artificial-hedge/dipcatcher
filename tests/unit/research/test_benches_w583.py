"""Wave-583 motivic-6 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w583 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_base_change": b.bench_motivic_base_change_family(),
        "six_op_motivic": b.bench_six_op_motivic_family(),
        "motivic_smooth": b.bench_motivic_smooth_family(),
        "motivic_proper": b.bench_motivic_proper_family(),
        "fulton_mclarty": b.bench_fulton_mclarty_family(),
        "motivic_homotopy2": b.bench_motivic_homotopy2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
