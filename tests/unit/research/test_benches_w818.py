"""Wave-818 integration adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w818 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ito_integral": b.bench_ito_integral_family(),
        "mart_meas": b.bench_mart_meas_family(),
        "vector_mart": b.bench_vector_mart_family(),
        "bounded_var": b.bench_bounded_var_family(),
        "stochastic_int2": b.bench_stochastic_int2_family(),
        "covariation": b.bench_covariation_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
