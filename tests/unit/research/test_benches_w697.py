"""Wave-697 spectral-AG-8 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w697 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "spectral_dvr": b.bench_spectral_dvr_family(),
        "spectral_noether": b.bench_spectral_noether_family(),
        "spectral_regular": b.bench_spectral_regular_family(),
        "spectral_dedekind": b.bench_spectral_dedekind_family(),
        "spectral_jacobson": b.bench_spectral_jacobson_family(),
        "spectral_excellent": b.bench_spectral_excellent_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
