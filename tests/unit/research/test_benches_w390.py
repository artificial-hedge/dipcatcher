"""Wave-390 design-theory adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w390 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "latin_trade": b.bench_latin_trade_family(),
        "steiner_system": b.bench_steiner_system_family(),
        "inc_structure": b.bench_inc_structure_family(),
        "orthogonal_array": b.bench_orthogonal_array_family(),
        "hadamard_matrix": b.bench_hadamard_matrix_family(),
        "finite_difference": b.bench_finite_difference_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
