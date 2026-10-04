"""Wave-872 preconditioner adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w872 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "jacobi_precond": b.bench_jacobi_precond_family(),
        "ilut_precond": b.bench_ilut_precond_family(),
        "ssor_precond": b.bench_ssor_precond_family(),
        "amg_precond": b.bench_amg_precond_family(),
        "ic_precond": b.bench_ic_precond_family(),
        "polynomial_precond": b.bench_polynomial_precond_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
