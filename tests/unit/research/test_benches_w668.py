"""Wave-668 motivic-15 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w668 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "norimotive3": b.bench_norimotive3_family(),
        "motivic_galois2": b.bench_motivic_galois2_family(),
        "tannakian_motive2": b.bench_tannakian_motive2_family(),
        "period_realization2": b.bench_period_realization2_family(),
        "beilinson_regulator2": b.bench_beilinson_regulator2_family(),
        "hodge_motive2": b.bench_hodge_motive2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
