"""Wave-611 category-8 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w611 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "pasting_diag": b.bench_pasting_diag_family(),
        "mate_dual": b.bench_mate_dual_family(),
        "whisker_comp": b.bench_whisker_comp_family(),
        "pseudo_naturality": b.bench_pseudo_naturality_family(),
        "two_adjoint": b.bench_two_adjoint_family(),
        "modification": b.bench_modification_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
