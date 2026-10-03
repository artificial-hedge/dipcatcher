"""Wave-459 order-theory-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w459 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fixed_points_ord": b.bench_fixed_points_ord_family(),
        "chain_cond": b.bench_chain_cond_family(),
        "scott_cpo": b.bench_scott_cpo_family(),
        "way_below": b.bench_way_below_family(),
        "galois_insertion": b.bench_galois_insertion_family(),
        "denotational": b.bench_denotational_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
