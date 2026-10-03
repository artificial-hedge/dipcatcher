"""Wave-844 asymptotic-analysis adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w844 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "asymptotic_series": b.bench_asymptotic_series_family(),
        "poincare_expansion": b.bench_poincare_expansion_family(),
        "steepest_descent": b.bench_steepest_descent_family(),
        "stationary_phase": b.bench_stationary_phase_family(),
        "borel_resum": b.bench_borel_resum_family(),
        "wkb_approx": b.bench_wkb_approx_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
