"""Wave-835 stochastic-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w835 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "poisson_voronoi": b.bench_poisson_voronoi_family(),
        "boolean_model": b.bench_boolean_model_family(),
        "germ_grain": b.bench_germ_grain_family(),
        "steiner_formula": b.bench_steiner_formula_family(),
        "miles_matheron": b.bench_miles_matheron_family(),
        "intrinsic_volumes": b.bench_intrinsic_volumes_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
