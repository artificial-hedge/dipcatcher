"""Wave-635 commutative-algebra-5 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w635 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "excellent_ring": b.bench_excellent_ring_family(),
        "zariski_main": b.bench_zariski_main_family(),
        "going_up": b.bench_going_up_family(),
        "lying_over": b.bench_lying_over_family(),
        "integral_closure2": b.bench_integral_closure2_family(),
        "weil_divisor2": b.bench_weil_divisor2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
