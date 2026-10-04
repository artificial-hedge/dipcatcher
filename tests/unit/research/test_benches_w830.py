"""Wave-830 concentration adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w830 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "azuma_ineq": b.bench_azuma_ineq_family(),
        "mcdiarmid_ineq": b.bench_mcdiarmid_ineq_family(),
        "talagrand_ineq": b.bench_talagrand_ineq_family(),
        "efron_stein": b.bench_efron_stein_family(),
        "bounded_diff": b.bench_bounded_diff_family(),
        "hoeffding_ineq": b.bench_hoeffding_ineq_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
