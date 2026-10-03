"""Wave-336 computability/model-theory adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w336 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "pr_functions": b.bench_pr_functions_family(),
        "turing_degrees": b.bench_turing_degrees_family(),
        "busy_beaver": b.bench_busy_beaver_family(),
        "ultraproduct": b.bench_ultraproduct_family(),
        "ramsey_theory": b.bench_ramsey_theory_family(),
        "compactness_lite": b.bench_compactness_lite_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
