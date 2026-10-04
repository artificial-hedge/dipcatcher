"""Wave-483 homotopy-10 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w483 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "steenrod_ops": b.bench_steenrod_ops_family(),
        "dyer_lashof": b.bench_dyer_lashof_family(),
        "bar_spec": b.bench_bar_spec_family(),
        "free_loop": b.bench_free_loop_family(),
        "sullivan_min": b.bench_sullivan_min_family(),
        "loop_functor": b.bench_loop_functor_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
