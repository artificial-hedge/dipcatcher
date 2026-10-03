"""Wave-437 p-adic cohomology adapter tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w437 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "crystalline_coh": b.bench_crystalline_coh_family(),
        "prismatic_coh": b.bench_prismatic_coh_family(),
        "etale_coh": b.bench_etale_coh_family(),
        "derham_coh": b.bench_derham_coh_family(),
        "frobenius_coh": b.bench_frobenius_coh_family(),
        "comparison_iso": b.bench_comparison_iso_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
