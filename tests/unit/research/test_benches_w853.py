"""Wave-853 spectral-element adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w853 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "sem_grid": b.bench_sem_grid_family(),
        "gll_nodes": b.bench_gll_nodes_family(),
        "spectral_element": b.bench_spectral_element_family(),
        "mortar_method": b.bench_mortar_method_family(),
        "tensor_product_sem": b.bench_tensor_product_sem_family(),
        "hp_refinement": b.bench_hp_refinement_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
