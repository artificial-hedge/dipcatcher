"""Wave-713 spectral-AG-10 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w713 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "spectral_semi": b.bench_spectral_semi_family(),
        "spectral_artin": b.bench_spectral_artin_family(),
        "spectral_gal": b.bench_spectral_gal_family(),
        "spectral_dirac": b.bench_spectral_dirac_family(),
        "derived_affine": b.bench_derived_affine_family(),
        "derived_projective": b.bench_derived_projective_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
