"""Wave-659 motivic-14 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w659 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "norimotive2": b.bench_norimotive2_family(),
        "motivic_tate2": b.bench_motivic_tate2_family(),
        "absolute_cohom": b.bench_absolute_cohom_family(),
        "motivic_weight": b.bench_motivic_weight_family(),
        "tate_triple": b.bench_tate_triple_family(),
        "motivic_pairing": b.bench_motivic_pairing_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
