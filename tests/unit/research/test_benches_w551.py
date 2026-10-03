"""Wave-551 foliation-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w551 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "foliation": b.bench_foliation_family(),
        "holonomy_grp": b.bench_holonomy_grp_family(),
        "godbillon_vey": b.bench_godbillon_vey_family(),
        "haefliger_struct": b.bench_haefliger_struct_family(),
        "novikov_thm": b.bench_novikov_thm_family(),
        "thurston_fol": b.bench_thurston_fol_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
