"""Wave-448 higher-topos adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w448 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "infty_topos": b.bench_infty_topos_family(),
        "univ_colimit": b.bench_univ_colimit_family(),
        "object_classif": b.bench_object_classif_family(),
        "trunc_modal": b.bench_trunc_modal_family(),
        "cohesive_top": b.bench_cohesive_top_family(),
        "hypercomplete": b.bench_hypercomplete_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
