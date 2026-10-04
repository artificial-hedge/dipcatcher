"""Wave-598 algebraic-K-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w598 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "borel_regulator": b.bench_borel_regulator_family(),
        "soul_elem": b.bench_soul_elem_family(),
        "lichtenbaum_k": b.bench_lichtenbaum_k_family(),
        "bloch_beilinson": b.bench_bloch_beilinson_family(),
        "etale_ktheory": b.bench_etale_ktheory_family(),
        "thh_trace": b.bench_thh_trace_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
