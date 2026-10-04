"""Wave-891 optimization adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w891 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "trust_region_dogleg": b.bench_trust_region_dogleg_family(),
        "bfgs_update": b.bench_bfgs_update_family(),
        "lebesgue_const": b.bench_lebesgue_const_family(),
        "iga_colloc": b.bench_iga_colloc_family(),
        "trimmed_cad": b.bench_trimmed_cad_family(),
        "newton_armijo": b.bench_newton_armijo_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
