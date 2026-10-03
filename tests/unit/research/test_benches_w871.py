"""Wave-871 Krylov-solver adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w871 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cg_solver": b.bench_cg_solver_family(),
        "gmres_solver": b.bench_gmres_solver_family(),
        "bicg_solver": b.bench_bicg_solver_family(),
        "arnoldi_eig": b.bench_arnoldi_eig_family(),
        "lanczos_eig": b.bench_lanczos_eig_family(),
        "lsqr_solver": b.bench_lsqr_solver_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
