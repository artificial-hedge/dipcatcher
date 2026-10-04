"""Wave-330 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w330 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "diff_logic": b.bench_diff_logic_family(),
        "array_theory": b.bench_array_theory_family(),
        "bv_ops": b.bench_bv_ops_family(),
        "dpllt": b.bench_dpllt_family(),
        "lia_branch": b.bench_lia_branch_family(),
        "mcsat_lite": b.bench_mcsat_lite_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
