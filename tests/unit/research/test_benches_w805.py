"""Wave-805 stochastic-calculus adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w805 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ito_isometry": b.bench_ito_isometry_family(),
        "stratonovich_conv": b.bench_stratonovich_conv_family(),
        "tanaka_meyer": b.bench_tanaka_meyer_family(),
        "follmer_strat": b.bench_follmer_strat_family(),
        "skorohod_lemma": b.bench_skorohod_lemma_family(),
        "doss_sussmann": b.bench_doss_sussmann_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
