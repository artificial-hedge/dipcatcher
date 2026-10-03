"""Wave-444 DAG-stacks adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w444 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derived_stack": b.bench_derived_stack_family(),
        "cotangent_cx": b.bench_cotangent_cx_family(),
        "geometric_stk": b.bench_geometric_stk_family(),
        "tannaka_rec": b.bench_tannaka_rec_family(),
        "quasi_smooth": b.bench_quasi_smooth_family(),
        "perf_stack": b.bench_perf_stack_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
