"""Wave-431 motivic-homotopy adapter tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w431 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "a1_homotopy": b.bench_a1_homotopy_family(),
        "motivic_sphere": b.bench_motivic_sphere_family(),
        "morel_degree": b.bench_morel_degree_family(),
        "voevodsky_motive": b.bench_voevodsky_motive_family(),
        "slice_filtration": b.bench_slice_filtration_family(),
        "milnor_operations": b.bench_milnor_operations_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
