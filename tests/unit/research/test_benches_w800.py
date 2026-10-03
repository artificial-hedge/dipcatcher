"""Wave-800 stochastic-vol adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w800 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "heston_model": b.bench_heston_model_family(),
        "bates_model": b.bench_bates_model_family(),
        "rough_heston": b.bench_rough_heston_family(),
        "sabr_model": b.bench_sabr_model_family(),
        "three_two_vol": b.bench_three_two_vol_family(),
        "scott_vol": b.bench_scott_vol_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
