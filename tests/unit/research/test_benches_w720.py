"""Wave-720 derived-dimension adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w720 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cluster_tilting": b.bench_cluster_tilting_family(),
        "derived_morita": b.bench_derived_morita_family(),
        "preprojective_alg": b.bench_preprojective_alg_family(),
        "categorical_entropy": b.bench_categorical_entropy_family(),
        "serre_dim": b.bench_serre_dim_family(),
        "rouquier_dim": b.bench_rouquier_dim_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
