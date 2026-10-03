"""Wave-682 motivic-18 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w682 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_tower": b.bench_motivic_tower_family(),
        "motivic_sphere3": b.bench_motivic_sphere3_family(),
        "motivic_etale": b.bench_motivic_etale_family(),
        "motivic_crystal": b.bench_motivic_crystal_family(),
        "motivic_prism": b.bench_motivic_prism_family(),
        "motivic_cycle": b.bench_motivic_cycle_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
