"""Wave-726 Galois-deformation adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w726 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "jetchev_skinner": b.bench_jetchev_skinner_family(),
        "wan_sss": b.bench_wan_sss_family(),
        "wiles_taylor": b.bench_wiles_taylor_family(),
        "diamond_taylor_wiles": b.bench_diamond_taylor_wiles_family(),
        "kisin_crystalline": b.bench_kisin_crystalline_family(),
        "mazur_deform": b.bench_mazur_deform_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
