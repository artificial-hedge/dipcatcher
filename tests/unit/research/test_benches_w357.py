"""Wave-357 differential-geometry-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w357 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "connection_form": b.bench_connection_form_family(),
        "parallel_transport": b.bench_parallel_transport_family(),
        "holonomy": b.bench_holonomy_family(),
        "gauss_bonnet": b.bench_gauss_bonnet_family(),
        "geodesic_eq": b.bench_geodesic_eq_family(),
        "sectional_curv": b.bench_sectional_curv_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
