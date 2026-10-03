"""Wave-654 category-11 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w654 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "compactly_generated": b.bench_compactly_generated_family(),
        "presentable_cat2": b.bench_presentable_cat2_family(),
        "accessible_cat2": b.bench_accessible_cat2_family(),
        "flat_monad": b.bench_flat_monad_family(),
        "locally_presentable": b.bench_locally_presentable_family(),
        "regular_cat2": b.bench_regular_cat2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
