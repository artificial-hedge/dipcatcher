"""Wave-634 category-9 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w634 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "icon_cat": b.bench_icon_cat_family(),
        "bicat2": b.bench_bicat2_family(),
        "vert_cat": b.bench_vert_cat_family(),
        "double_lim": b.bench_double_lim_family(),
        "two_transform": b.bench_two_transform_family(),
        "cat_3cell": b.bench_cat_3cell_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
