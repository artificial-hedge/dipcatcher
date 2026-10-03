"""Wave-847 perturbation-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w847 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "regular_perturbation": b.bench_regular_perturbation_family(),
        "singular_perturbation": b.bench_singular_perturbation_family(),
        "matched_asymptotic": b.bench_matched_asymptotic_family(),
        "multiple_scales": b.bench_multiple_scales_family(),
        "lindstedt_poincare": b.bench_lindstedt_poincare_family(),
        "boundary_layer": b.bench_boundary_layer_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
