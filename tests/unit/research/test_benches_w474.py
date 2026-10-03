"""Wave-474 TQFT-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w474 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "reshet_turaev": b.bench_reshet_turaev_family(),
        "khovanov": b.bench_khovanov_family(),
        "heegaard_floer": b.bench_heegaard_floer_family(),
        "cobordism_hyp": b.bench_cobordism_hyp_family(),
        "modular_cat": b.bench_modular_cat_family(),
        "topological_order": b.bench_topological_order_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
