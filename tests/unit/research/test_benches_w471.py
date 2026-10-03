"""Wave-471 p-adic-geometry-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w471 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "perfectoid2": b.bench_perfectoid2_family(),
        "diamond_geo": b.bench_diamond_geo_family(),
        "integral_padic": b.bench_integral_padic_family(),
        "breuil_kisin": b.bench_breuil_kisin_family(),
        "banach_colmez": b.bench_banach_colmez_family(),
        "drinfeld_tower": b.bench_drinfeld_tower_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
