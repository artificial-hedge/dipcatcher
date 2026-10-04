"""Wave-612 commutative-algebra-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w612 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "regular_ring": b.bench_regular_ring_family(),
        "gorenstein_ring": b.bench_gorenstein_ring_family(),
        "normal_ring": b.bench_normal_ring_family(),
        "factorial_ring": b.bench_factorial_ring_family(),
        "jacobson_ring": b.bench_jacobson_ring_family(),
        "discrete_valuation": b.bench_discrete_valuation_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
