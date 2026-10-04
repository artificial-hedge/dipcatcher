"""Wave-684 motivic-19 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w684 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_weight2": b.bench_motivic_weight2_family(),
        "motivic_infinite": b.bench_motivic_infinite_family(),
        "motivic_suslin": b.bench_motivic_suslin_family(),
        "motivic_frequency": b.bench_motivic_frequency_family(),
        "motivic_wit": b.bench_motivic_wit_family(),
        "motivic_base2": b.bench_motivic_base2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
