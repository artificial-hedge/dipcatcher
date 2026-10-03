"""Wave-391 category-theory-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w391 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "monoidal_cat": b.bench_monoidal_cat_family(),
        "closed_cat": b.bench_closed_cat_family(),
        "presheaf": b.bench_presheaf_family(),
        "kan_extension": b.bench_kan_extension_family(),
        "distributor": b.bench_distributor_family(),
        "equivalence_cat": b.bench_equivalence_cat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
