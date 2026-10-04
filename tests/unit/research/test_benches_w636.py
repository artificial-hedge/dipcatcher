"""Wave-636 tensor-category-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w636 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "sylleptic": b.bench_sylleptic_family(),
        "haagerup_sub": b.bench_haagerup_sub_family(),
        "ek_subfactor": b.bench_ek_subfactor_family(),
        "gyro_cat": b.bench_gyro_cat_family(),
        "yang_lee_cat": b.bench_yang_lee_cat_family(),
        "sovereign_cat": b.bench_sovereign_cat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
