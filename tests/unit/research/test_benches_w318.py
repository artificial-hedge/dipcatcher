"""Wave-318 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w318 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "bidirectional_tc": b.bench_bidirectional_tc_family(),
        "nbe_eval": b.bench_nbe_eval_family(),
        "dep_types": b.bench_dep_types_family(),
        "unify_meta": b.bench_unify_meta_family(),
        "proof_kernel": b.bench_proof_kernel_family(),
        "tactic_engine": b.bench_tactic_engine_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
