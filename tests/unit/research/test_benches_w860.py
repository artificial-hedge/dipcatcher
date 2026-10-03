"""Wave-860 isogeometric/immersed-methods adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w860 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "iso_geom": b.bench_iso_geom_family(),
        "nurbs_elem": b.bench_nurbs_elem_family(),
        "xfem": b.bench_xfem_family(),
        "immersed_boundary": b.bench_immersed_boundary_family(),
        "cut_cell": b.bench_cut_cell_family(),
        "fictitious_domain": b.bench_fictitious_domain_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
