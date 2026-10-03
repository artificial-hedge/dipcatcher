"""Wave-384 group-theory-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w384 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hall_subgroup": b.bench_hall_subgroup_family(),
        "transfer_hom": b.bench_transfer_hom_family(),
        "schur_multiplier": b.bench_schur_multiplier_family(),
        "aut_group": b.bench_aut_group_family(),
        "composition_series": b.bench_composition_series_family(),
        "permutation_poly": b.bench_permutation_poly_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
