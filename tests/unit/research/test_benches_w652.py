"""Wave-652 motivic-12 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w652 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "strict_motive": b.bench_strict_motive_family(),
        "sheaf_motive": b.bench_sheaf_motive_family(),
        "numerical_motive": b.bench_numerical_motive_family(),
        "asymptotic_motive": b.bench_asymptotic_motive_family(),
        "exponential_motive": b.bench_exponential_motive_family(),
        "log_motive": b.bench_log_motive_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
