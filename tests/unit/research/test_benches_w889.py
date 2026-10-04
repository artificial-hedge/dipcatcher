"""Wave-889 FE-basis adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w889 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "quadrilateral_basis": b.bench_quadrilateral_basis_family(),
        "hexahedral_basis": b.bench_hexahedral_basis_family(),
        "chebyshev_u": b.bench_chebyshev_u_family(),
        "walsh_table": b.bench_walsh_table_family(),
        "epsilon_algo": b.bench_epsilon_algo_family(),
        "spline_theory": b.bench_spline_theory_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
