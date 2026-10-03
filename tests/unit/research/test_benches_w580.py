"""Wave-580 enumerative-combinatorics adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w580 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "species_theory": b.bench_species_theory_family(),
        "cycle_index": b.bench_cycle_index_family(),
        "lagrange_inversion": b.bench_lagrange_inversion_family(),
        "transfer_matrix": b.bench_transfer_matrix_family(),
        "matrix_tree": b.bench_matrix_tree_family(),
        "exponential_gf": b.bench_exponential_gf_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
