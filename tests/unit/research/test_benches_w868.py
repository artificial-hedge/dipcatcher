"""Wave-868 continuation/homotopy adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w868 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "arc_continuation": b.bench_arc_continuation_family(),
        "pseudo_arclength": b.bench_pseudo_arclength_family(),
        "deflation_method": b.bench_deflation_method_family(),
        "bifurcation_track": b.bench_bifurcation_track_family(),
        "homotopy_solver": b.bench_homotopy_solver_family(),
        "davidenko_ode": b.bench_davidenko_ode_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
