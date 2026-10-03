"""Wave-888 RBF/basis adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w888 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "thin_plate_spline": b.bench_thin_plate_spline_family(),
        "polyharmonic_rbf": b.bench_polyharmonic_rbf_family(),
        "trefethen_diff": b.bench_trefethen_diff_family(),
        "galerkin_projection": b.bench_galerkin_projection_family(),
        "periodic_spline": b.bench_periodic_spline_family(),
        "zernike_poly": b.bench_zernike_poly_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
