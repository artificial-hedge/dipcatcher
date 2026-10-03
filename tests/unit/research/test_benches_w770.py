"""Wave-770 extreme-value adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w770 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gumbel_domain": b.bench_gumbel_domain_family(),
        "weibull_domain": b.bench_weibull_domain_family(),
        "frechet_domain": b.bench_frechet_domain_family(),
        "peak_over": b.bench_peak_over_family(),
        "hill_est": b.bench_hill_est_family(),
        "pickands_est": b.bench_pickands_est_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
