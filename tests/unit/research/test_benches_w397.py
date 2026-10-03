"""Wave-397 stochastic-analysis-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w397 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ost_calcul": b.bench_ost_calcul_family(),
        "tanaka": b.bench_tanaka_family(),
        "bessel3": b.bench_bessel3_family(),
        "reflect_bm": b.bench_reflect_bm_family(),
        "occupation_bm": b.bench_occupation_bm_family(),
        "h_transform": b.bench_h_transform_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
