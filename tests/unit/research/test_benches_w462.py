"""Wave-462 infinity-topos-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w462 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "n_localic": b.bench_n_localic_family(),
        "shape_theory": b.bench_shape_theory_family(),
        "descent_cond": b.bench_descent_cond_family(),
        "lex_reflect": b.bench_lex_reflect_family(),
        "cartesian_fib2": b.bench_cartesian_fib2_family(),
        "cohesive_struct": b.bench_cohesive_struct_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
