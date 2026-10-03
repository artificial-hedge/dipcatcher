"""Wave-359 harmonic-analysis adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w359 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "plancherel": b.bench_plancherel_family(),
        "poisson_summation": b.bench_poisson_summation_family(),
        "fejer_kernel": b.bench_fejer_kernel_family(),
        "uncertainty": b.bench_uncertainty_family(),
        "fourier_multiplier": b.bench_fourier_multiplier_family(),
        "sobolev_embed": b.bench_sobolev_embed_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
