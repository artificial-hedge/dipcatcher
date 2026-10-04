"""Wave-578 arithmetic-geometry-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w578 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "global_height": b.bench_global_height_family(),
        "bogomolov_conj": b.bench_bogomolov_conj_family(),
        "equidistribution_thm": b.bench_equidistribution_thm_family(),
        "canonical_height": b.bench_canonical_height_family(),
        "nevanlinna_th": b.bench_nevanlinna_th_family(),
        "vojta_conj": b.bench_vojta_conj_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
