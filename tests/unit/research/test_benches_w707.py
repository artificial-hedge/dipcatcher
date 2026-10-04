"""Wave-707 spectral-AG-9 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w707 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "spectral_prime": b.bench_spectral_prime_family(),
        "spectral_residue": b.bench_spectral_residue_family(),
        "spectral_level": b.bench_spectral_level_family(),
        "spectral_polynomial2": b.bench_spectral_polynomial2_family(),
        "spectral_coord": b.bench_spectral_coord_family(),
        "spectral_ext_field": b.bench_spectral_ext_field_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
