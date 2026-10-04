"""Wave-716 triangulated-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w716 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "exceptional_coll": b.bench_exceptional_coll_family(),
        "spherical_functor": b.bench_spherical_functor_family(),
        "serre_functor": b.bench_serre_functor_family(),
        "sod_decomp": b.bench_sod_decomp_family(),
        "fourier_mukai": b.bench_fourier_mukai_family(),
        "semi_orthogonal": b.bench_semi_orthogonal_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
