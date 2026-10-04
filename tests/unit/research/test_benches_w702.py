"""Wave-702 motivic-23 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w702 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_spark": b.bench_motivic_spark_family(),
        "motivic_fundamental": b.bench_motivic_fundamental_family(),
        "motivic_hochschild": b.bench_motivic_hochschild_family(),
        "motivic_field": b.bench_motivic_field_family(),
        "motivic_degree": b.bench_motivic_degree_family(),
        "motivic_diagonal": b.bench_motivic_diagonal_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
