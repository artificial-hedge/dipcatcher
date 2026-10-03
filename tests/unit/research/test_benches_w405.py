"""Wave-405 derived-categories adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w405 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derived_functor2": b.bench_derived_functor2_family(),
        "triangulated": b.bench_triangulated_family(),
        "bounded_complex": b.bench_bounded_complex_family(),
        "mapping_cone_tri": b.bench_mapping_cone_tri_family(),
        "koszul_dual": b.bench_koszul_dual_family(),
        "t_structure": b.bench_t_structure_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
