"""Wave-739 Brownian-map-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w739 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "caraceni_curien": b.bench_caraceni_curien_family(),
        "bonzom_combe": b.bench_bonzom_combe_family(),
        "mullin_bijection": b.bench_mullin_bijection_family(),
        "bernardi_bijection": b.bench_bernardi_bijection_family(),
        "schaeffer_bijection": b.bench_schaeffer_bijection_family(),
        "bouttier_guiter": b.bench_bouttier_guiter_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
