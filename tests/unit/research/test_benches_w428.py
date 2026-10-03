"""Wave-428 deformation-theory adapter tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w428 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "deformation_functor": b.bench_deformation_functor_family(),
        "schlessinger": b.bench_schlessinger_family(),
        "tangent_space_def": b.bench_tangent_space_def_family(),
        "obstruction_theory": b.bench_obstruction_theory_family(),
        "versal_deformation": b.bench_versal_deformation_family(),
        "maurer_cartan": b.bench_maurer_cartan_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
