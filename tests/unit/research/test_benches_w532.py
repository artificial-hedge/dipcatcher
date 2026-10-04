"""Wave-532 fractal-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w532 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hausdorff_dim": b.bench_hausdorff_dim_family(),
        "box_counting": b.bench_box_counting_family(),
        "self_similar": b.bench_self_similar_family(),
        "iterated_function": b.bench_iterated_function_family(),
        "frostman": b.bench_frostman_family(),
        "multifractal_formal": b.bench_multifractal_formal_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
