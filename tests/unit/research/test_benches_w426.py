"""Wave-426 homotopy-7 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w426 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "model_category": b.bench_model_category_family(),
        "quillen_adj": b.bench_quillen_adj_family(),
        "simplicial_set": b.bench_simplicial_set_family(),
        "infinity_cat": b.bench_infinity_cat_family(),
        "derived_alg": b.bench_derived_alg_family(),
        "stable_cat": b.bench_stable_cat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
