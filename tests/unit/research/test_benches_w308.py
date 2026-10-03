"""Wave-308 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w308 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "reflectivity_synth": b.bench_reflectivity_synth_family(),
        "gassmann_sub": b.bench_gassmann_sub_family(),
        "spectral_decomp": b.bench_spectral_decomp_family(),
        "semblance_scan": b.bench_semblance_scan_family(),
        "gardner_relation": b.bench_gardner_relation_family(),
        "vz_raytrace": b.bench_vz_raytrace_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
