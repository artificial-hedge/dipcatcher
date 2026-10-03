"""Wave-700 category-20 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w700 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cat_pushout": b.bench_cat_pushout_family(),
        "cat_span": b.bench_cat_span_family(),
        "cat_lax": b.bench_cat_lax_family(),
        "cat_street": b.bench_cat_street_family(),
        "cat_size": b.bench_cat_size_family(),
        "cat_total": b.bench_cat_total_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
