"""Wave-381 algebraic-geometry-6 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w381 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "grothendieck_grp": b.bench_grothendieck_grp_family(),
        "chow_ring": b.bench_chow_ring_family(),
        "gysin": b.bench_gysin_family(),
        "toric_variety": b.bench_toric_variety_family(),
        "proj_morph": b.bench_proj_morph_family(),
        "ample_test": b.bench_ample_test_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
