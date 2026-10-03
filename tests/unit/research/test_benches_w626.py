"""Wave-626 stacks-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w626 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "algebraic_stack2": b.bench_algebraic_stack2_family(),
        "artin_stack": b.bench_artin_stack_family(),
        "quotient_stack2": b.bench_quotient_stack2_family(),
        "stacky_point": b.bench_stacky_point_family(),
        "orbifold_stack": b.bench_orbifold_stack_family(),
        "gerbe_cohomology": b.bench_gerbe_cohomology_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
