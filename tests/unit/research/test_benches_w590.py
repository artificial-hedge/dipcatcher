"""Wave-590 algebraic-K-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w590 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "quillen_plus": b.bench_quillen_plus_family(),
        "gersten_ss": b.bench_gersten_ss_family(),
        "loday_k": b.bench_loday_k_family(),
        "volodin_k": b.bench_volodin_k_family(),
        "suslin_k": b.bench_suslin_k_family(),
        "bloch_k": b.bench_bloch_k_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
