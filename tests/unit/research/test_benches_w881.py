"""Wave-881 Krylov-solver adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w881 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "minres_solver": b.bench_minres_solver_family(),
        "cgs_solver": b.bench_cgs_solver_family(),
        "tfqmr": b.bench_tfqmr_family(),
        "qmr_solver": b.bench_qmr_solver_family(),
        "bicgstab2": b.bench_bicgstab2_family(),
        "block_cg": b.bench_block_cg_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
