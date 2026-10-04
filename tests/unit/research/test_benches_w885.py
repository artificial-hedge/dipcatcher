"""Wave-885 flux/asymptotics adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w885 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ldg_flux": b.bench_ldg_flux_family(),
        "entropy_stable_dg": b.bench_entropy_stable_dg_family(),
        "wkb_turning": b.bench_wkb_turning_family(),
        "averaging_method": b.bench_averaging_method_family(),
        "laplace_method": b.bench_laplace_method_family(),
        "hyperasymptotic": b.bench_hyperasymptotic_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
