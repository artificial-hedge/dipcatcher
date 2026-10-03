"""Wave-591 motivic-7 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w591 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "friedlander_voev": b.bench_friedlander_voev_family(),
        "motivic_eilenberg": b.bench_motivic_eilenberg_family(),
        "motivic_zeta": b.bench_motivic_zeta_family(),
        "motivic_purity": b.bench_motivic_purity_family(),
        "motivic_descent": b.bench_motivic_descent_family(),
        "motivic_invert": b.bench_motivic_invert_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
