"""Wave-831 empirical-process adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w831 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "entropy_integral": b.bench_entropy_integral_family(),
        "uniform_clt": b.bench_uniform_clt_family(),
        "symmetrization": b.bench_symmetrization_family(),
        "rademacher_cplx": b.bench_rademacher_cplx_family(),
        "covering_number": b.bench_covering_number_family(),
        "metric_entropy": b.bench_metric_entropy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
