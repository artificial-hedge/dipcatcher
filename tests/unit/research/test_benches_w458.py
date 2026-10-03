"""Wave-458 matroid-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w458 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "transversal_mat": b.bench_transversal_mat_family(),
        "matroid_rep": b.bench_matroid_rep_family(),
        "tutte_poly": b.bench_tutte_poly_family(),
        "matroid_minor": b.bench_matroid_minor_family(),
        "regular_mat": b.bench_regular_mat_family(),
        "delta_matroid": b.bench_delta_matroid_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
