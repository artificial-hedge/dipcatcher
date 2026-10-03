"""Wave-774 random-walk adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w774 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "sparc_rw": b.bench_sparc_rw_family(),
        "spitzer_rw": b.bench_spitzer_rw_family(),
        "fluctuation_rw": b.bench_fluctuation_rw_family(),
        "ladder_epoch": b.bench_ladder_epoch_family(),
        "wiener_hopf_rw": b.bench_wiener_hopf_rw_family(),
        "maxwell_rw": b.bench_maxwell_rw_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
