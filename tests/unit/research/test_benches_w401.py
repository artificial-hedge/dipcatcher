"""Wave-401 proof-theory-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w401 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "herbrand_thm": b.bench_herbrand_thm_family(),
        "interp_equality": b.bench_interp_equality_family(),
        "cut_elim_seq": b.bench_cut_elim_seq_family(),
        "finitary_induct": b.bench_finitary_induct_family(),
        "hilbert_system": b.bench_hilbert_system_family(),
        "reverse_math": b.bench_reverse_math_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
