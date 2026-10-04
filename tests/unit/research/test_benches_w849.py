"""Wave-849 finite-volume adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w849 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fdm_grid": b.bench_fdm_grid_family(),
        "compact_scheme": b.bench_compact_scheme_family(),
        "crank_nicholson2": b.bench_crank_nicholson2_family(),
        "upwind_scheme": b.bench_upwind_scheme_family(),
        "muscl_reconstruct": b.bench_muscl_reconstruct_family(),
        "flux_splitting": b.bench_flux_splitting_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
