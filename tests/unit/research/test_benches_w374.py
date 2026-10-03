"""Wave-374 model-theory-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w374 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "quantifier_elim": b.bench_quantifier_elim_family(),
        "realize_types": b.bench_realize_types_family(),
        "omega_categoricity": b.bench_omega_categoricity_family(),
        "acl_closure": b.bench_acl_closure_family(),
        "morley_rank": b.bench_morley_rank_family(),
        "vocab_interp": b.bench_vocab_interp_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
