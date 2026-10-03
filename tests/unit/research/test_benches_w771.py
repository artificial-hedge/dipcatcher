"""Wave-771 copula adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w771 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "copula_gauss": b.bench_copula_gauss_family(),
        "copula_t": b.bench_copula_t_family(),
        "clayton_copula": b.bench_clayton_copula_family(),
        "gumbel_copula": b.bench_gumbel_copula_family(),
        "frank_copula": b.bench_frank_copula_family(),
        "joe_copula": b.bench_joe_copula_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
