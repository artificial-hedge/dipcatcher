"""Wave-627 condensed-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w627 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "discrete_liquid": b.bench_discrete_liquid_family(),
        "smith_project": b.bench_smith_project_family(),
        "condensed_ring": b.bench_condensed_ring_family(),
        "liquid_ring": b.bench_liquid_ring_family(),
        "scholze_trace": b.bench_scholze_trace_family(),
        "condensed_coh": b.bench_condensed_coh_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
