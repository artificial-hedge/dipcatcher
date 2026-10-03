"""Wave-335 category-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w335 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fin_limit": b.bench_fin_limit_family(),
        "subobject_classifier": b.bench_subobject_classifier_family(),
        "exponential_obj": b.bench_exponential_obj_family(),
        "yoneda_embed": b.bench_yoneda_embed_family(),
        "adjoint_check": b.bench_adjoint_check_family(),
        "cat_colimit": b.bench_cat_colimit_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
