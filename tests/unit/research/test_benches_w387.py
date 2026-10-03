"""Wave-387 probability-4 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w387 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "uniform_integrability": b.bench_uniform_integrability_family(),
        "vitali_conv": b.bench_vitali_conv_family(),
        "ldp_theory": b.bench_ldp_theory_family(),
        "concentration_ineq": b.bench_concentration_ineq_family(),
        "kolmogorov_01": b.bench_kolmogorov_01_family(),
        "prokhorov_metric": b.bench_prokhorov_metric_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
