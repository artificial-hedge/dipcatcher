"""Wave-638 algebraic-K-7 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w638 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "grayson_s": b.bench_grayson_s_family(),
        "karoubi_v2": b.bench_karoubi_v2_family(),
        "vorst_descent": b.bench_vorst_descent_family(),
        "quillen_ldev": b.bench_quillen_ldev_family(),
        "fundamental_cat": b.bench_fundamental_cat_family(),
        "seg_street": b.bench_seg_street_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
