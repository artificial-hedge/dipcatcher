"""Wave-346 number-theory-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w346 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "quadratic_recip": b.bench_quadratic_recip_family(),
        "elliptic_curve": b.bench_elliptic_curve_family(),
        "p_adic_val": b.bench_p_adic_val_family(),
        "cohomology_cup": b.bench_cohomology_cup_family(),
        "koszul_complex": b.bench_koszul_complex_family(),
        "mayer_vietoris": b.bench_mayer_vietoris_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
