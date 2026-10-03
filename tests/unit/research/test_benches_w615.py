"""Wave-615 motivic-9 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w615 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_adams": b.bench_motivic_adams_family(),
        "motivic_classifying": b.bench_motivic_classifying_family(),
        "tate_object": b.bench_tate_object_family(),
        "motivic_dg": b.bench_motivic_dg_family(),
        "mori_bir": b.bench_mori_bir_family(),
        "extremal_ray": b.bench_extremal_ray_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
