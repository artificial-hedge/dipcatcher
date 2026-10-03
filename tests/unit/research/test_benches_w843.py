"""Wave-843 spline-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w843 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "b_spline": b.bench_b_spline_family(),
        "de_boor": b.bench_de_boor_family(),
        "cardinal_spline": b.bench_cardinal_spline_family(),
        "knot_insertion": b.bench_knot_insertion_family(),
        "blossoming": b.bench_blossoming_family(),
        "box_spline": b.bench_box_spline_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
