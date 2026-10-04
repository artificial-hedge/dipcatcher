"""Wave-392 computability adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w392 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "mu_recursion": b.bench_mu_recursion_family(),
        "primitive_recursion": b.bench_primitive_recursion_family(),
        "diagonal_lemma": b.bench_diagonal_lemma_family(),
        "arithmetization": b.bench_arithmetization_family(),
        "fixed_point_combinator": b.bench_fixed_point_combinator_family(),
        "kleene_normal": b.bench_kleene_normal_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
