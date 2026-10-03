"""Wave-680 spectral-AG-6 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w680 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "spectral_perfect": b.bench_spectral_perfect_family(),
        "spectral_smooth2": b.bench_spectral_smooth2_family(),
        "spectral_etale2": b.bench_spectral_etale2_family(),
        "spectral_abelian": b.bench_spectral_abelian_family(),
        "spectral_crystal": b.bench_spectral_crystal_family(),
        "spectral_proper": b.bench_spectral_proper_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
