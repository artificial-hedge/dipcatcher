"""Wave-629 deformations-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w629 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "deform_functor2": b.bench_deform_functor2_family(),
        "tangent_def": b.bench_tangent_def_family(),
        "rim_deform": b.bench_rim_deform_family(),
        "small_ext": b.bench_small_ext_family(),
        "hull_deform": b.bench_hull_deform_family(),
        "artinian_alg": b.bench_artinian_alg_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
