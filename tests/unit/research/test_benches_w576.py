"""Wave-576 free-probability adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w576 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "free_prob": b.bench_free_prob_family(),
        "r_transform": b.bench_r_transform_family(),
        "s_transform": b.bench_s_transform_family(),
        "free_convolution": b.bench_free_convolution_family(),
        "voiculescu_thm": b.bench_voiculescu_thm_family(),
        "operator_valued": b.bench_operator_valued_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
