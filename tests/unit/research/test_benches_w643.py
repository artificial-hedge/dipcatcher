"""Wave-643 homotopy-18 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w643 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "toda_smith": b.bench_toda_smith_family(),
        "mahowald_inv": b.bench_mahowald_inv_family(),
        "calc_tower": b.bench_calc_tower_family(),
        "goodwillie_deriv": b.bench_goodwillie_deriv_family(),
        "snaith_split": b.bench_snaith_split_family(),
        "kervaire_inv": b.bench_kervaire_inv_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
