"""Wave-783 stochastic-order adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w783 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "semi_mart": b.bench_semi_mart_family(),
        "predictable_bracket": b.bench_predictable_bracket_family(),
        "cramer_wold": b.bench_cramer_wold_family(),
        "stricker_thm": b.bench_stricker_thm_family(),
        "likelihood_order": b.bench_likelihood_order_family(),
        "hazard_order": b.bench_hazard_order_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
