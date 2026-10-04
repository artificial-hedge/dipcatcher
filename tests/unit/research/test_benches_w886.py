"""Wave-886 QMC/tensor adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w886 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "faure_seq": b.bench_faure_seq_family(),
        "importance_mc": b.bench_importance_mc_family(),
        "gauss_hermite": b.bench_gauss_hermite_family(),
        "gauss_laguerre": b.bench_gauss_laguerre_family(),
        "tensor_train": b.bench_tensor_train_family(),
        "hiot_decomp": b.bench_hiot_decomp_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
