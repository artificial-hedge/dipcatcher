"""Wave-438 condensed-mathematics adapter tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w438 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "condensed_set": b.bench_condensed_set_family(),
        "solid_group": b.bench_solid_group_family(),
        "liquid_group": b.bench_liquid_group_family(),
        "proetale_site": b.bench_proetale_site_family(),
        "light_condensed": b.bench_light_condensed_family(),
        "analytic_ring": b.bench_analytic_ring_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
