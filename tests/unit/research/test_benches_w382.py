"""Wave-382 model-theory-4 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w382 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "stone_duality": b.bench_stone_duality_family(),
        "saturation_test": b.bench_saturation_test_family(),
        "omitting_types": b.bench_omitting_types_family(),
        "indiscernibles": b.bench_indiscernibles_family(),
        "stability_spec": b.bench_stability_spec_family(),
        "back_forth": b.bench_back_forth_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
