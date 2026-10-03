"""Wave-690 homotopy-27 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w690 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "homotopy_sheaf": b.bench_homotopy_sheaf_family(),
        "homotopy_model": b.bench_homotopy_model_family(),
        "stable_monoid": b.bench_stable_monoid_family(),
        "stable_group": b.bench_stable_group_family(),
        "stable_module": b.bench_stable_module_family(),
        "stable_algebra": b.bench_stable_algebra_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
