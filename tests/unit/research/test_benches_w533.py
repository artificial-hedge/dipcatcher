"""Wave-533 geometric-measure-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w533 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "rectifiability": b.bench_rectifiability_family(),
        "tangent_measure": b.bench_tangent_measure_family(),
        "density_thm": b.bench_density_thm_family(),
        "marstrand": b.bench_marstrand_family(),
        "besicovitch": b.bench_besicovitch_family(),
        "preiss_rect": b.bench_preiss_rect_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
