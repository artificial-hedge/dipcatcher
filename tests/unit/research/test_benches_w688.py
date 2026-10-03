"""Wave-688 spectral-AG-7 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w688 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "spectral_field": b.bench_spectral_field_family(),
        "spectral_lattice": b.bench_spectral_lattice_family(),
        "spectral_filtration": b.bench_spectral_filtration_family(),
        "spectral_cellular": b.bench_spectral_cellular_family(),
        "spectral_cohomological": b.bench_spectral_cohomological_family(),
        "spectral_finite": b.bench_spectral_finite_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
