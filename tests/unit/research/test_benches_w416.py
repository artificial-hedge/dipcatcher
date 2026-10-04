"""Wave-416 model-theory-6 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w416 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "decidable_theory": b.bench_decidable_theory_family(),
        "indiscernible_seq": b.bench_indiscernible_seq_family(),
        "saturated_model": b.bench_saturated_model_family(),
        "omitting_prime": b.bench_omitting_prime_family(),
        "interpol_thm": b.bench_interpol_thm_family(),
        "definable_set": b.bench_definable_set_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
