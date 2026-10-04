"""Wave-325 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w325 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "borrow_check": b.bench_borrow_check_family(),
        "lifetime_outlives": b.bench_lifetime_outlives_family(),
        "linear_use": b.bench_linear_use_family(),
        "escape_region": b.bench_escape_region_family(),
        "capability_perm": b.bench_capability_perm_family(),
        "refinement_liquid": b.bench_refinement_liquid_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
