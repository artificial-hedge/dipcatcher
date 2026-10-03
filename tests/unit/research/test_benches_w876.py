"""Wave-876 nonlinear-solver adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w876 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "moore_penrose": b.bench_moore_penrose_family(),
        "landweber_iter": b.bench_landweber_iter_family(),
        "conjugate_grad_ls": b.bench_conjugate_grad_ls_family(),
        "gauss_newton": b.bench_gauss_newton_family(),
        "levenberg_marq": b.bench_levenberg_marq_family(),
        "anderson_mixing": b.bench_anderson_mixing_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
