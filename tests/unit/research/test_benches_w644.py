"""Wave-644 algebraic-K-8 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w644 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "s_multicat": b.bench_s_multicat_family(),
        "allday_k": b.bench_allday_k_family(),
        "residue_k": b.bench_residue_k_family(),
        "suslin_wagoner": b.bench_suslin_wagoner_family(),
        "weibel_nil": b.bench_weibel_nil_family(),
        "hall_alg": b.bench_hall_alg_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
