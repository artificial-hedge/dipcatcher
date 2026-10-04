"""Wave-617 algebraic-K-6 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w617 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gillet_thomason": b.bench_gillet_thomason_family(),
        "khomo_k": b.bench_khomo_k_family(),
        "k_theory4": b.bench_k_theory4_family(),
        "gersen_suslin": b.bench_gersen_suslin_family(),
        "berrick_k": b.bench_berrick_k_family(),
        "hermitian_quillen": b.bench_hermitian_quillen_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
