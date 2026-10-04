"""Wave-658 motivic-13 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w658 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_galois": b.bench_motivic_galois_family(),
        "tannakian_motive": b.bench_tannakian_motive_family(),
        "period_realization": b.bench_period_realization_family(),
        "beilinson_regulator": b.bench_beilinson_regulator_family(),
        "hodge_motive": b.bench_hodge_motive_family(),
        "f_motive": b.bench_f_motive_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
