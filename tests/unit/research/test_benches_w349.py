"""Wave-349 algebraic-geometry adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w349 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "zariski_topo": b.bench_zariski_topo_family(),
        "projective_plane": b.bench_projective_plane_family(),
        "bezout_bezout": b.bench_bezout_bezout_family(),
        "variety_dim": b.bench_variety_dim_family(),
        "monomial_ideal": b.bench_monomial_ideal_family(),
        "hilbert_poly": b.bench_hilbert_poly_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
