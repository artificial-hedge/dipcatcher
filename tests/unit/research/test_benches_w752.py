"""Wave-752 UST/LERW adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w752 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "wilson_ust": b.bench_wilson_ust_family(),
        "lawler_lerw": b.bench_lawler_lerw_family(),
        "benjamini_ust": b.bench_benjamini_ust_family(),
        "kirchhoff_matrix": b.bench_kirchhoff_matrix_family(),
        "pemantle_ust": b.bench_pemantle_ust_family(),
        "schramm_lerw": b.bench_schramm_lerw_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
