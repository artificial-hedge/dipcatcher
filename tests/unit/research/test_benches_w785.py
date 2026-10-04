"""Wave-785 semimartingale-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w785 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "usual_cond": b.bench_usual_cond_family(),
        "dolean_mart": b.bench_dolean_mart_family(),
        "strong_sol": b.bench_strong_sol_family(),
        "local_mart2": b.bench_local_mart2_family(),
        "follmer_mart": b.bench_follmer_mart_family(),
        "protter_ito": b.bench_protter_ito_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
