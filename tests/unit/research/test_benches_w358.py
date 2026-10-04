"""Wave-358 functional-analysis-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w358 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "open_mapping": b.bench_open_mapping_family(),
        "uniform_bounded": b.bench_uniform_bounded_family(),
        "weak_convergence": b.bench_weak_convergence_family(),
        "banach_alaoglu": b.bench_banach_alaoglu_family(),
        "reflexive_space": b.bench_reflexive_space_family(),
        "closed_graph": b.bench_closed_graph_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
