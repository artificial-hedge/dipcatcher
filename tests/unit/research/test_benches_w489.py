"""Wave-489 group-theory-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w489 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "building_toy": b.bench_building_toy_family(),
        "coxeter_grp": b.bench_coxeter_grp_family(),
        "bn_pair": b.bench_bn_pair_family(),
        "braid_grp": b.bench_braid_grp_family(),
        "hecke_bm": b.bench_hecke_bm_family(),
        "parabolic_grp": b.bench_parabolic_grp_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
