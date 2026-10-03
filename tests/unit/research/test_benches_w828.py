"""Wave-828 projection/section adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w828 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "projection_theorem": b.bench_projection_theorem_family(),
        "uniform_section": b.bench_uniform_section_family(),
        "dellacherie_section": b.bench_dellacherie_section_family(),
        "cross_section": b.bench_cross_section_family(),
        "maharam_lift": b.bench_maharam_lift_family(),
        "von_neumann_sel": b.bench_von_neumann_sel_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
