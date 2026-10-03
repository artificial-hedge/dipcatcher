"""Wave-594 etale adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w594 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "etale_homotopy": b.bench_etale_homotopy_family(),
        "pro_etale": b.bench_pro_etale_family(),
        "etale_fund": b.bench_etale_fund_family(),
        "galois_cat": b.bench_galois_cat_family(),
        "artin_neighborhood": b.bench_artin_neighborhood_family(),
        "shapiro_lemma": b.bench_shapiro_lemma_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
