"""Wave-719 Calabi-Yau/Gorenstein adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w719 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "calabi_yau_tri": b.bench_calabi_yau_tri_family(),
        "d_calabi_yau": b.bench_d_calabi_yau_family(),
        "gorenstein_proj": b.bench_gorenstein_proj_family(),
        "frobenius_cat": b.bench_frobenius_cat_family(),
        "stable_category": b.bench_stable_category_family(),
        "orbit_category": b.bench_orbit_category_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
