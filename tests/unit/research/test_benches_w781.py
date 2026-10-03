"""Wave-781 regeneration/Khinchin adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w781 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "karlin_mcg": b.bench_karlin_mcg_family(),
        "keilson_stieltjes": b.bench_keilson_stieltjes_family(),
        "palm_khinchin": b.bench_palm_khinchin_family(),
        "regen_proc": b.bench_regen_proc_family(),
        "wold_proc": b.bench_wold_proc_family(),
        "korolyuk": b.bench_korolyuk_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
