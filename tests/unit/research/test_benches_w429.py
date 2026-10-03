"""Wave-429 algebraic-K-theory adapter tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w429 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "k0_group": b.bench_k0_group_family(),
        "k1_group": b.bench_k1_group_family(),
        "milnor_k2": b.bench_milnor_k2_family(),
        "quillen_q": b.bench_quillen_q_family(),
        "k_theory_spec": b.bench_k_theory_spec_family(),
        "bass_heller_swan": b.bench_bass_heller_swan_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
