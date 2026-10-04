"""Wave-679 motivic-17 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w679 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_thh": b.bench_motivic_thh_family(),
        "motivic_realization": b.bench_motivic_realization_family(),
        "etale_motive": b.bench_etale_motive_family(),
        "relative_motive": b.bench_relative_motive_family(),
        "absolute_motive": b.bench_absolute_motive_family(),
        "motivic_heart": b.bench_motivic_heart_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
