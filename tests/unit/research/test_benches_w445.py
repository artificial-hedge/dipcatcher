"""Wave-445 six-functor adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w445 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "six_functors": b.bench_six_functors_family(),
        "base_change": b.bench_base_change_family(),
        "projection_frm": b.bench_projection_frm_family(),
        "verdier_dual": b.bench_verdier_dual_family(),
        "constructible": b.bench_constructible_family(),
        "perverse_sh": b.bench_perverse_sh_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
