"""Wave-848 finite-element adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w848 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fem_assembly": b.bench_fem_assembly_family(),
        "isoparametric_map": b.bench_isoparametric_map_family(),
        "quadrature_rules": b.bench_quadrature_rules_family(),
        "triangular_basis": b.bench_triangular_basis_family(),
        "edge_elements": b.bench_edge_elements_family(),
        "dof_management": b.bench_dof_management_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
