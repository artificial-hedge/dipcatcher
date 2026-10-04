"""Wave-606 algebraic-K-5 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w606 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "connective_k": b.bench_connective_k_family(),
        "higher_k": b.bench_higher_k_family(),
        "k_spectrum": b.bench_k_spectrum_family(),
        "nil_k": b.bench_nil_k_family(),
        "karoubi_k": b.bench_karoubi_k_family(),
        "pedersen_weibel": b.bench_pedersen_weibel_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
