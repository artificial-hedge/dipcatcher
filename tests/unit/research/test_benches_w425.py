"""Wave-425 stacks/moduli adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w425 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "moduli_stack": b.bench_moduli_stack_family(),
        "stacky_curve": b.bench_stacky_curve_family(),
        "coarse_space": b.bench_coarse_space_family(),
        "quotient_stack": b.bench_quotient_stack_family(),
        "gerbe_toy": b.bench_gerbe_toy_family(),
        "stack_morph": b.bench_stack_morph_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
