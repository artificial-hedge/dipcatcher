"""Wave-504 Steenrod/cohomology-operations adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w504 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "steenrod_algebra": b.bench_steenrod_algebra_family(),
        "adem_relations": b.bench_adem_relations_family(),
        "serre_cartan": b.bench_serre_cartan_family(),
        "unstable_modules": b.bench_unstable_modules_family(),
        "lambda_algebra": b.bench_lambda_algebra_family(),
        "bar_resolution": b.bench_bar_resolution_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
