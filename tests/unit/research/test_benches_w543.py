"""Wave-543 Riemann-surfaces adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w543 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "riemann_surface": b.bench_riemann_surface_family(),
        "branched_cover": b.bench_branched_cover_family(),
        "abel_jacobi": b.bench_abel_jacobi_family(),
        "riemann_hurwitz": b.bench_riemann_hurwitz_family(),
        "fuchsian_group": b.bench_fuchsian_group_family(),
        "teichmuller_space": b.bench_teichmuller_space_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
