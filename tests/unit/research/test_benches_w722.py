"""Wave-722 special-values adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w722 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "period_poly": b.bench_period_poly_family(),
        "specialization_motive": b.bench_specialization_motive_family(),
        "borel_motivic": b.bench_borel_motivic_family(),
        "zagier_polylog": b.bench_zagier_polylog_family(),
        "deligne_period": b.bench_deligne_period_family(),
        "motivic_multiple_zeta": b.bench_motivic_multiple_zeta_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
