"""Wave-864 stochastic-Galerkin/UQ adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w864 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "stochastic_galerkin": b.bench_stochastic_galerkin_family(),
        "poly_chaos_uq": b.bench_poly_chaos_uq_family(),
        "intrusive_pce": b.bench_intrusive_pce_family(),
        "nonintrusive_pce": b.bench_nonintrusive_pce_family(),
        "stochastic_colloc": b.bench_stochastic_colloc_family(),
        "stochastic_fem": b.bench_stochastic_fem_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
