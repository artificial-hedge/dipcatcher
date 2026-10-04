"""Wave-793 stochastic-expansion adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w793 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "wong_zakai": b.bench_wong_zakai_family(),
        "stochastic_taylor": b.bench_stochastic_taylor_family(),
        "milstein_scheme": b.bench_milstein_scheme_family(),
        "wagner_platen": b.bench_wagner_platen_family(),
        "cubature_wiener": b.bench_cubature_wiener_family(),
        "rough_vol2": b.bench_rough_vol2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
