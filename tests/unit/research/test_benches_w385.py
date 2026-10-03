"""Wave-385 commutative-algebra-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w385 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hilbert_samuel": b.bench_hilbert_samuel_family(),
        "krull_dim": b.bench_krull_dim_family(),
        "noether_normal": b.bench_noether_normal_family(),
        "primary_decomp": b.bench_primary_decomp_family(),
        "completion_ring": b.bench_completion_ring_family(),
        "dimension_fiber": b.bench_dimension_fiber_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
