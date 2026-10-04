"""Wave-640 spectral-AG-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w640 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "spectral_group": b.bench_spectral_group_family(),
        "azure_space": b.bench_azure_space_family(),
        "spectral_scheme3": b.bench_spectral_scheme3_family(),
        "spectral_smooth": b.bench_spectral_smooth_family(),
        "spectral_etale": b.bench_spectral_etale_family(),
        "elliptic_cohom2": b.bench_elliptic_cohom2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
