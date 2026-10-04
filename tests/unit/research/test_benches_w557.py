"""Wave-557 3-manifold adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w557 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "heegaard_splitting": b.bench_heegaard_splitting_family(),
        "dehn_surgery": b.bench_dehn_surgery_family(),
        "sutured_mfd": b.bench_sutured_mfd_family(),
        "taut_foliation": b.bench_taut_foliation_family(),
        "thin_position": b.bench_thin_position_family(),
        "normal_surface": b.bench_normal_surface_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
