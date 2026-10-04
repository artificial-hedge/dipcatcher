"""Wave-792 BSDE adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w792 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "bsde_solver": b.bench_bsde_solver_family(),
        "fbsde_markov": b.bench_fbsde_markov_family(),
        "backward_sde": b.bench_backward_sde_family(),
        "pardoux_peng": b.bench_pardoux_peng_family(),
        "reflected_bsde": b.bench_reflected_bsde_family(),
        "second_order_bsde": b.bench_second_order_bsde_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
