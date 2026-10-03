"""Wave-400 algebraic-geometry-7 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w400 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "etale_cover": b.bench_etale_cover_family(),
        "jacobian_toy": b.bench_jacobian_toy_family(),
        "hom_stack_toy": b.bench_hom_stack_toy_family(),
        "seesaw_theorem": b.bench_seesaw_theorem_family(),
        "picard_variety": b.bench_picard_variety_family(),
        "dual_ab_var": b.bench_dual_ab_var_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
