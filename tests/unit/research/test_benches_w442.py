"""Wave-442 model-categories-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w442 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cofibrant_rep": b.bench_cofibrant_rep_family(),
        "quillen_equiv": b.bench_quillen_equiv_family(),
        "monoidal_model": b.bench_monoidal_model_family(),
        "enriched_model": b.bench_enriched_model_family(),
        "reedy_model": b.bench_reedy_model_family(),
        "localization_mc": b.bench_localization_mc_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
