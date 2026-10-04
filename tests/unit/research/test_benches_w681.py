"""Wave-681 homotopy-25 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w681 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "homotopy_lift": b.bench_homotopy_lift_family(),
        "homotopy_orbit": b.bench_homotopy_orbit_family(),
        "homotopy_fixed": b.bench_homotopy_fixed_family(),
        "stable_operad": b.bench_stable_operad_family(),
        "homotopy_factor": b.bench_homotopy_factor_family(),
        "stable_sheaf": b.bench_stable_sheaf_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
