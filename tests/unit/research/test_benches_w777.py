"""Wave-777 heavy-traffic adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w777 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fluid_limit": b.bench_fluid_limit_family(),
        "heavy_traffic": b.bench_heavy_traffic_family(),
        "diffusion_approx": b.bench_diffusion_approx_family(),
        "kingman_bound": b.bench_kingman_bound_family(),
        "halfin_whitt": b.bench_halfin_whitt_family(),
        "qed_regime": b.bench_qed_regime_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
