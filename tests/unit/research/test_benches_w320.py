"""Wave-320 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w320 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "interval_analysis": b.bench_interval_analysis_family(),
        "sign_domain": b.bench_sign_domain_family(),
        "zone_dbm": b.bench_zone_dbm_family(),
        "affine_karr": b.bench_affine_karr_family(),
        "chaotic_widen": b.bench_chaotic_widen_family(),
        "andersen_pta": b.bench_andersen_pta_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
