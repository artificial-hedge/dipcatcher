"""Wave-833 functional-limit adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w833 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fclt_invariance": b.bench_fclt_invariance_family(),
        "donsker_invariance": b.bench_donsker_invariance_family(),
        "martingale_fclt": b.bench_martingale_fclt_family(),
        "stable_limit": b.bench_stable_limit_family(),
        "brownian_approx": b.bench_brownian_approx_family(),
        "strassen_flln": b.bench_strassen_flln_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
