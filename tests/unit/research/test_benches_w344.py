"""Wave-344 group-theory-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w344 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "sylow_theorems": b.bench_sylow_theorems_family(),
        "group_presentation": b.bench_group_presentation_family(),
        "burnside_lemma": b.bench_burnside_lemma_family(),
        "free_group": b.bench_free_group_family(),
        "conjugacy_classes": b.bench_conjugacy_classes_family(),
        "cayley_graph": b.bench_cayley_graph_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
