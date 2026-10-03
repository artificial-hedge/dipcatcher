"""Wave-633 motivic-11 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w633 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_coho2": b.bench_motivic_coho2_family(),
        "cone_theorem": b.bench_cone_theorem_family(),
        "motivic_landweber": b.bench_motivic_landweber_family(),
        "motivic_abelian": b.bench_motivic_abelian_family(),
        "motivic_compact": b.bench_motivic_compact_family(),
        "contr_rational": b.bench_contr_rational_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
