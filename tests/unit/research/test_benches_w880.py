"""Wave-880 quadrature/cubature adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w880 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gq_adaptive": b.bench_gq_adaptive_family(),
        "adaptive_quad2": b.bench_adaptive_quad2_family(),
        "pod_deim": b.bench_pod_deim_family(),
        "empirical_interp": b.bench_empirical_interp_family(),
        "cubature_rule": b.bench_cubature_rule_family(),
        "tensor_interp": b.bench_tensor_interp_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
