"""Wave-689 motivic-20 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w689 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_jouanolou": b.bench_motivic_jouanolou_family(),
        "motivic_infinite2": b.bench_motivic_infinite2_family(),
        "motivic_ext": b.bench_motivic_ext_family(),
        "motivic_norm": b.bench_motivic_norm_family(),
        "motivic_ramified": b.bench_motivic_ramified_family(),
        "motivic_euler": b.bench_motivic_euler_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
