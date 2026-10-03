"""Wave-393 algebraic-topology-4 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w393 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "eilenberg_steenrod": b.bench_eilenberg_steenrod_family(),
        "cap_product": b.bench_cap_product_family(),
        "thom_isom": b.bench_thom_isom_family(),
        "serre_class": b.bench_serre_class_family(),
        "obstruction_toy": b.bench_obstruction_toy_family(),
        "k_theory": b.bench_k_theory_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
