"""Wave-674 spectral-AG-5 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w674 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derived_k3": b.bench_derived_k3_family(),
        "spectral_gm": b.bench_spectral_gm_family(),
        "analytic_spec": b.bench_analytic_spec_family(),
        "graded_spec": b.bench_graded_spec_family(),
        "equivariant_spec": b.bench_equivariant_spec_family(),
        "spectral_curve": b.bench_spectral_curve_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
