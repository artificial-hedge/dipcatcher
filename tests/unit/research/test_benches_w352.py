"""Wave-352 Lie-theory adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w352 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cartan_matrix": b.bench_cartan_matrix_family(),
        "weyl_group_a2": b.bench_weyl_group_a2_family(),
        "killing_form": b.bench_killing_form_family(),
        "root_lattice_a2": b.bench_root_lattice_a2_family(),
        "sl2_structure": b.bench_sl2_structure_family(),
        "su2_algebra": b.bench_su2_algebra_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
