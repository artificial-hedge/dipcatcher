"""Wave-665 homotopy-23 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w665 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cohen_moore2": b.bench_cohen_moore2_family(),
        "whitehead_product": b.bench_whitehead_product_family(),
        "homotopy_decomp": b.bench_homotopy_decomp_family(),
        "kervaire_inv2": b.bench_kervaire_inv2_family(),
        "unstable_vn": b.bench_unstable_vn_family(),
        "moore_space2": b.bench_moore_space2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
