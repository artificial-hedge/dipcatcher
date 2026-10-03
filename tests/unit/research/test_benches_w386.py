"""Wave-386 proof-theory-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w386 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "sequent_calculus": b.bench_sequent_calculus_family(),
        "natural_ded": b.bench_natural_ded_family(),
        "godel_incomp": b.bench_godel_incomp_family(),
        "interp_proof": b.bench_interp_proof_family(),
        "proof_complexity": b.bench_proof_complexity_family(),
        "modal_completeness": b.bench_modal_completeness_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
