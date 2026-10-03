"""Wave-826 maximal-inequality adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w826 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "doob_ineq": b.bench_doob_ineq_family(),
        "max_ineq": b.bench_max_ineq_family(),
        "bj_ineq": b.bench_bj_ineq_family(),
        "kolmogorov_ineq": b.bench_kolmogorov_ineq_family(),
        "etemadi_ineq": b.bench_etemadi_ineq_family(),
        "levy_ineq": b.bench_levy_ineq_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
