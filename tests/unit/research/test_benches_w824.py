"""Wave-824 stochastic-order adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w824 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "usual_stoch_order": b.bench_usual_stoch_order_family(),
        "first_order_dom": b.bench_first_order_dom_family(),
        "second_order_dom": b.bench_second_order_dom_family(),
        "convex_order": b.bench_convex_order_family(),
        "hazard_rate_order": b.bench_hazard_rate_order_family(),
        "supermodular_order": b.bench_supermodular_order_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
