"""Wave-321 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w321 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "three_valued_logic": b.bench_three_valued_logic_family(),
        "shape_graph": b.bench_shape_graph_family(),
        "separation_logic": b.bench_separation_logic_family(),
        "context_pta": b.bench_context_pta_family(),
        "recency_abstraction": b.bench_recency_abstraction_family(),
        "interproc_summary": b.bench_interproc_summary_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
