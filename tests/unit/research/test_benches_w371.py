"""Wave-371 differential-topology adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w371 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "morse_theory": b.bench_morse_theory_family(),
        "transversality": b.bench_transversality_family(),
        "regular_value": b.bench_regular_value_family(),
        "degree_mod2": b.bench_degree_mod2_family(),
        "handle_decomp": b.bench_handle_decomp_family(),
        "poincare_hopf": b.bench_poincare_hopf_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
