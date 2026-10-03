"""Wave-663 higher-algebra-5 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w663 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "e2_algebra": b.bench_e2_algebra_family(),
        "dunn_additivity": b.bench_dunn_additivity_family(),
        "tensor_factorization": b.bench_tensor_factorization_family(),
        "swiss_cheese2": b.bench_swiss_cheese2_family(),
        "mckay_correspond": b.bench_mckay_correspond_family(),
        "khovanov_2": b.bench_khovanov_2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
