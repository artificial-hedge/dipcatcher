"""Wave-642 prismatic-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w642 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "prismatic_f": b.bench_prismatic_f_family(),
        "bhatt_scholze": b.bench_bhatt_scholze_family(),
        "q_crystal": b.bench_q_crystal_family(),
        "prismatic_dieudonne": b.bench_prismatic_dieudonne_family(),
        "q_prism": b.bench_q_prism_family(),
        "derived_prism": b.bench_derived_prism_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
