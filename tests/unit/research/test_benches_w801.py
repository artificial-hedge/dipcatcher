"""Wave-801 rough-volatility adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w801 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fractional_heston": b.bench_fractional_heston_family(),
        "rough_bergomi": b.bench_rough_bergomi_family(),
        "rough_sabr": b.bench_rough_sabr_family(),
        "rough_variance": b.bench_rough_variance_family(),
        "volterra_sde": b.bench_volterra_sde_family(),
        "multifactor_rough": b.bench_multifactor_rough_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
