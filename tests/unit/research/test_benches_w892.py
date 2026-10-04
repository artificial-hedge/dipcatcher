"""Wave-892 adaptive-mesh adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w892 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "space_time_adapt": b.bench_space_time_adapt_family(),
        "greedy_marking": b.bench_greedy_marking_family(),
        "form_analysis": b.bench_form_analysis_family(),
        "hp_adaptive": b.bench_hp_adaptive_family(),
        "wavelet_adapt": b.bench_wavelet_adapt_family(),
        "residual_marking": b.bench_residual_marking_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
