"""Wave-672 category-14 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w672 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "tannakian_cat": b.bench_tannakian_cat_family(),
        "super_cat": b.bench_super_cat_family(),
        "perverse_cat": b.bench_perverse_cat_family(),
        "smashing_cat": b.bench_smashing_cat_family(),
        "compact_cat": b.bench_compact_cat_family(),
        "monoidal_derived": b.bench_monoidal_derived_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
