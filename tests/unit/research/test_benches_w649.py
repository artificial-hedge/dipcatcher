"""Wave-649 prismatic-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w649 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "prism_site2": b.bench_prism_site2_family(),
        "cartier_prism": b.bench_cartier_prism_family(),
        "breuil_prism": b.bench_breuil_prism_family(),
        "filtered_prism": b.bench_filtered_prism_family(),
        "frobenius_prism": b.bench_frobenius_prism_family(),
        "stacky_prism": b.bench_stacky_prism_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
