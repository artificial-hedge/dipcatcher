"""Wave-418 probability-5 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w418 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "weak_law": b.bench_weak_law_family(),
        "strong_lln": b.bench_strong_lln_family(),
        "clt_classic": b.bench_clt_classic_family(),
        "borel_cantelli": b.bench_borel_cantelli_family(),
        "dominated_conv": b.bench_dominated_conv_family(),
        "uniform_lln": b.bench_uniform_lln_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
