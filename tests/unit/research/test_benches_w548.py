"""Wave-548 Floer-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w548 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "floer_homology": b.bench_floer_homology_family(),
        "knot_floer": b.bench_knot_floer_family(),
        "instanton_floer": b.bench_instanton_floer_family(),
        "monopole_floer": b.bench_monopole_floer_family(),
        "lagrangian_floer": b.bench_lagrangian_floer_family(),
        "fukaya_cat": b.bench_fukaya_cat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
