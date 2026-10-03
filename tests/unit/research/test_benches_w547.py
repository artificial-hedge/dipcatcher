"""Wave-547 4-manifold adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w547 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "four_mfd": b.bench_four_mfd_family(),
        "donaldson_thm": b.bench_donaldson_thm_family(),
        "seiberg_witten": b.bench_seiberg_witten_family(),
        "exotic_r4": b.bench_exotic_r4_family(),
        "intersection_form": b.bench_intersection_form_family(),
        "freedman_thm": b.bench_freedman_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
