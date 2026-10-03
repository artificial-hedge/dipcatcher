"""Wave-379 matroid-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w379 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "matroid_axioms": b.bench_matroid_axioms_family(),
        "greedy_matroid": b.bench_greedy_matroid_family(),
        "matroid_intersect": b.bench_matroid_intersect_family(),
        "dual_matroid": b.bench_dual_matroid_family(),
        "matroid_union": b.bench_matroid_union_family(),
        "represented_matroid": b.bench_represented_matroid_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
