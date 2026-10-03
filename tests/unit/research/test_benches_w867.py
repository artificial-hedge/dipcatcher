"""Wave-867 inverse-problem adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w867 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "tikhonov_reg": b.bench_tikhonov_reg_family(),
        "morozov_dp": b.bench_morozov_dp_family(),
        "l_curve_opt": b.bench_l_curve_opt_family(),
        "iter_regularize": b.bench_iter_regularize_family(),
        "tv_denoise": b.bench_tv_denoise_family(),
        "bayes_inverse": b.bench_bayes_inverse_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
