"""Wave-410 set-theory-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w410 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "forcing2": b.bench_forcing2_family(),
        "inner_model": b.bench_inner_model_family(),
        "descriptive3": b.bench_descriptive3_family(),
        "recursion3": b.bench_recursion3_family(),
        "proof_mining": b.bench_proof_mining_family(),
        "ordinal_notation": b.bench_ordinal_notation_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
