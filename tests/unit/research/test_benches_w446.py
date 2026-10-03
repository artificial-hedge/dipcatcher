"""Wave-446 Goodwillie-calculus adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w446 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "goodwillie_tower": b.bench_goodwillie_tower_family(),
        "excisive_fn": b.bench_excisive_fn_family(),
        "linearization": b.bench_linearization_family(),
        "deriv_layer": b.bench_deriv_layer_family(),
        "calc_converge": b.bench_calc_converge_family(),
        "orth_calc": b.bench_orth_calc_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
