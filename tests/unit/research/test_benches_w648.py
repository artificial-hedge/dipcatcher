"""Wave-648 algebraic-K-10 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w648 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "kodaira_k": b.bench_kodaira_k_family(),
        "lindenstrauss_k": b.bench_lindenstrauss_k_family(),
        "tsukada_k": b.bench_tsukada_k_family(),
        "guin_k": b.bench_guin_k_family(),
        "dupont_k": b.bench_dupont_k_family(),
        "suslin_k2": b.bench_suslin_k2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
