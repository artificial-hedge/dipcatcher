"""Wave-673 derived-geometry-6 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w673 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derived_cohom": b.bench_derived_cohom_family(),
        "spectral_deformation2": b.bench_spectral_deformation2_family(),
        "virtual_class2": b.bench_virtual_class2_family(),
        "derived_intersection": b.bench_derived_intersection_family(),
        "derived_fiber2": b.bench_derived_fiber2_family(),
        "relative_trace": b.bench_relative_trace_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
