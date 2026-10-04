"""Wave-436 infinity-categories-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w436 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "complete_seg": b.bench_complete_seg_family(),
        "cartesian_fib": b.bench_cartesian_fib_family(),
        "straightening": b.bench_straightening_family(),
        "presentable_cat": b.bench_presentable_cat_family(),
        "adjoint_functor": b.bench_adjoint_functor_family(),
        "bousfield_loc": b.bench_bousfield_loc_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
