"""Wave-711 homotopy-31 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w711 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "homotopy_sheaf2": b.bench_homotopy_sheaf2_family(),
        "stable_inf_cat": b.bench_stable_inf_cat_family(),
        "homotopy_stable4": b.bench_homotopy_stable4_family(),
        "homotopy_local": b.bench_homotopy_local_family(),
        "stable_sheaf2": b.bench_stable_sheaf2_family(),
        "stable_coalgebra": b.bench_stable_coalgebra_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
